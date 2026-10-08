/* * MIT License
 *
 * © Squey, 2026
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

// DuckDB hands the scan the WHERE clause and removes the operator that would
// have applied it. Accepting a filter is therefore a commitment, not a hint:
// anything taken and not applied widens the result, silently.
//
// The scan answers what pvcop settles by dictionary lookup -- equality and
// membership -- and converts the rest back into an expression it evaluates per
// chunk. Every case below is checked against pvcop read directly, so a filter
// mapped to the wrong column, dropped, or applied twice shows up as a count
// that does not match.
//
// The cases are chosen for what they exercise rather than for variety: a filter
// on a column that is not the first, the two routes combined in one query, a
// filter over the restricted table, and one selective enough that whole blocks
// come back empty -- an empty chunk being how a thread says it is done, that
// last one would truncate a scan that returned it.

#include <squey/PVDuckDBQuery.h>
#include <squey/PVSelection.h>
#include <squey/PVSource.h>
#include <squey/PVView.h>

#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVNraw.h>

#include <pvcop/db/array.h>

#include <string>
#include <vector>

#include "common.h"

const std::string filename = TEST_FOLDER "/picviz/axes_types_discovery.csv";
const std::string fileformat = TEST_FOLDER "/picviz/axes_types_discovery.csv.format";

// Two text axes far apart in the format, so a filter keyed to the wrong column
// cannot pass by coincidence.
constexpr PVCol IPV4_COL(10);
constexpr char IPV4_COL_NAME[] = "ipv4";
constexpr PVCol STRING_COL(13);
constexpr char STRING_COL_NAME[] = "string";

/**
 * Count the rows a predicate keeps, reading pvcop directly. This is the oracle
 * every SQL result below is compared against.
 */
template <class F>
static size_t count_if_rows(const PVRush::PVNraw& nraw, F&& predicate)
{
	size_t count = 0;
	for (size_t row = 0; row < nraw.row_count(); ++row) {
		count += static_cast<size_t>(predicate(row));
	}
	return count;
}

int main()
{
	pvtest::TestEnv env(filename, fileformat, 1, pvtest::ProcessUntil::View);
	Squey::PVView* view = env.root.current_view();
	const PVRush::PVNraw& nraw = view->get_rushnraw_parent();
	const size_t row_count = nraw.row_count();

	Squey::PVDuckDBQuery query(view->get_parent<Squey::PVSource>());

	const std::string str_col = Squey::PVDuckDBQuery::quote_identifier(STRING_COL_NAME);
	const std::string ip_col = Squey::PVDuckDBQuery::quote_identifier(IPV4_COL_NAME);
	PV_VALID(query.column_names()[STRING_COL + 1], std::string(STRING_COL_NAME));
	PV_VALID(query.column_names()[IPV4_COL + 1], std::string(IPV4_COL_NAME));

	const pvcop::db::array& strings = nraw.column(STRING_COL);
	const pvcop::db::array& ips = nraw.column(IPV4_COL);

	Squey::PVSelection out(row_count);

	// --- Equality on a late column: taken by pvcop ----------------------------
	// Selective enough that most blocks hold no matching row at all.
	{
		const std::string value = strings.at(row_count / 3);
		const size_t expected =
		    count_if_rows(nraw, [&](size_t row) { return strings.at(row) == value; });
		PV_ASSERT_VALID(expected > 0, "expected", expected);

		query.select("SELECT rowid FROM layers WHERE " + str_col + " = '" + value + "'", out);
		PV_VALID(out.bit_count(), expected);
	}

	// --- Membership -----------------------------------------------------------
	{
		const std::string a = strings.at(0);
		const std::string b = strings.at(row_count / 2);
		const std::string c = strings.at(row_count - 1);
		const size_t expected = count_if_rows(nraw, [&](size_t row) {
			const std::string v = strings.at(row);
			return v == a || v == b || v == c;
		});

		query.select("SELECT rowid FROM layers WHERE " + str_col + " IN ('" + a + "', '" + b +
		                 "', '" + c + "')",
		             out);
		PV_VALID(out.bit_count(), expected);
	}

	// --- Two filters on two columns: both taken, composed ---------------------
	{
		const std::string value = ips.at(row_count / 5);
		const size_t expected = count_if_rows(nraw, [&](size_t row) {
			return ips.at(row) == value && strings.at(row) == strings.at(row_count / 5);
		});
		PV_ASSERT_VALID(expected > 0, "expected", expected);

		query.select("SELECT rowid FROM layers WHERE " + ip_col + " = '" + value + "' AND " +
		                 str_col + " = '" + strings.at(row_count / 5) + "'",
		             out);
		PV_VALID(out.bit_count(), expected);
	}

	// --- One route each, in the same query ------------------------------------
	// The equality goes to pvcop, the pattern stays with DuckDB. Both have to
	// apply: dropping either would widen the result.
	{
		const std::string value = strings.at(row_count / 7);
		const size_t expected = count_if_rows(nraw, [&](size_t row) {
			return strings.at(row) == value && ips.at(row).find('1') != std::string::npos;
		});

		query.select("SELECT rowid FROM layers WHERE " + str_col + " = '" + value + "' AND " +
		                 ip_col + " LIKE '%1%'",
		             out);
		PV_VALID(out.bit_count(), expected);
	}

	// --- A filter that keeps nothing -------------------------------------------
	// Every chunk comes back empty, which is also how a thread reports it is
	// done: a scan that hands such a chunk back stops early instead.
	{
		query.select("SELECT rowid FROM layers WHERE " + str_col + " = 'no such value'", out);
		PV_VALID(out.bit_count(), size_t(0));
	}

	// --- Composed with an input selection ---------------------------------------
	// The restricted table already narrows the scan; a pushed filter narrows it
	// further, and the two have to intersect rather than replace one another.
	{
		const std::string value = strings.at(row_count / 3);

		Squey::PVSelection in(row_count);
		in.select_none();
		size_t expected = 0;
		for (size_t row = 0; row < row_count; row += 2) {
			in.set_line(PVRow(row), true);
			expected += static_cast<size_t>(strings.at(row) == value);
		}

		query.select("SELECT rowid FROM selection WHERE " + str_col + " = '" + value + "'", in, out);
		PV_VALID(out.bit_count(), expected);

		// The same predicate over every row finds at least as many, and the
		// restricted answer has to be exactly the intersection.
		Squey::PVSelection whole(row_count);
		query.select("SELECT rowid FROM layers WHERE " + str_col + " = '" + value + "'", whole);
		Squey::PVSelection intersection = whole;
		intersection &= in;
		PV_ASSERT_VALID((out ^ intersection).is_empty(), "restricted", out.bit_count(),
		                "intersection", intersection.bit_count());
	}

	// --- A filter on rowid ------------------------------------------------------
	// rowid is not an nraw column, so pvcop has nothing to say about it: this
	// goes to the expression route with no column behind it.
	{
		query.select("SELECT rowid FROM layers WHERE rowid < 100", out);
		PV_VALID(out.bit_count(), size_t(100));
	}

	// --- Negation ---------------------------------------------------------------
	// Not an equality, so DuckDB keeps it. Checked because the scan must not
	// mistake it for one.
	{
		const std::string value = strings.at(0);
		const size_t expected =
		    count_if_rows(nraw, [&](size_t row) { return strings.at(row) != value; });

		query.select("SELECT rowid FROM layers WHERE " + str_col + " <> '" + value + "'", out);
		PV_VALID(out.bit_count(), expected);
	}

	return 0;
}
