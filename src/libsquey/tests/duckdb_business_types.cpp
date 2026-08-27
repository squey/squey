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
//
// A business type is stored as the integer whose order is its own order -- an
// address as a UINTEGER -- which is what lets the scan hand DuckDB a pointer
// into the column and what makes ORDER BY the address order. It also leaves a
// query reading 3232235777 where the listing shows 192.168.1.1, so each such
// type carries a pair of conversions.
//
// Both are checked against pvcop's own formatter, which is what the listing
// draws: the reading one must produce exactly that text, and the writing one
// must select exactly the rows holding it. Anchoring them to each other would
// only prove they agree, which two conversions written from the same wrong
// assumption also do.
//
// Run once per type, taking the file, format, column and conversion name from
// argv, the way the pushdown test does.

#include <squey/PVDuckDBQuery.h>
#include <squey/PVSelection.h>
#include <squey/PVSource.h>
#include <squey/PVView.h>

#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVNraw.h>

#include <pvcop/db/array.h>

#include <algorithm>
#include <string>
#include <vector>

#include "common.h"

int main(int argc, char** argv)
{
	PV_ASSERT_VALID(argc >= 5, "usage",
	                std::string("<file> <format> <column_name> <conversion_name>"));
	const std::string file = argv[1];
	const std::string format = argv[2];
	const std::string column_name = argv[3];
	const std::string conversion = argv[4];

	pvtest::TestEnv env(file, format, 1, pvtest::ProcessUntil::View);
	Squey::PVView* view = env.root.current_view();
	const PVRush::PVNraw& nraw = view->get_rushnraw_parent();
	const size_t row_count = nraw.row_count();

	Squey::PVDuckDBQuery query(view->get_parent<Squey::PVSource>());

	const std::vector<std::string> names = query.column_names();
	size_t index = names.size();
	for (size_t i = 1; i < names.size(); ++i) {
		if (names[i] == column_name) {
			index = i;
			break;
		}
	}
	PV_ASSERT_VALID(index < names.size(), "column not found", column_name);

	// The conversion is named after the axis type, which is the rule that makes
	// it findable at all: what the format editor calls the axis is what a query
	// writes.
	PV_VALID(query.column_axis_types()[index], conversion);

	const PVCol col(static_cast<PVCol::value_type>(index - 1));
	const pvcop::db::array& array = nraw.column(col);
	const std::string quoted = Squey::PVDuckDBQuery::quote_identifier(column_name);

	// The column must be exposed as the integer, not as text: the conversions
	// exist because of that, and a fallback to VARCHAR would make them pointless
	// while every assertion below still passed.
	PV_ASSERT_VALID(query.column_types()[index] != "VARCHAR", "the column fell back to text",
	                query.column_types()[index]);

	const size_t probes = std::min<size_t>(32, row_count);
	Squey::PVSelection out(row_count);

	for (size_t row = 0; row < probes; ++row) {
		const std::string written = array.at(row);

		// --- Reading: what the conversion shows is what the listing draws ---------
		const auto shown = query.run_tabular("SELECT " + conversion + "_text(" + quoted +
		                                         ") FROM layers WHERE rowid = " + std::to_string(row),
		                                     nullptr, 1);
		PV_VALID(shown.rows.size(), size_t(1));
		PV_VALID(shown.rows[0][0], written);

		// --- Writing: the literal selects the rows holding that value -----------
		// The comparison stays on the stored integer -- the conversion applies to
		// the constant -- which is the point of having it.
		size_t expected = 0;
		for (size_t r = 0; r < row_count; ++r) {
			expected += static_cast<size_t>(array.at(r) == written);
		}
		PV_ASSERT_VALID(expected > 0, "expected", expected);

		query.select("SELECT rowid FROM layers WHERE " + quoted + " = " + conversion + "('" +
		                 written + "')",
		             out);
		PV_ASSERT_VALID(out.bit_count() == expected, "row", row, "value", written, "sql",
		                out.bit_count(), "pvcop", expected);
	}

	// --- The two directions compose over the whole column ---------------------
	// Applied to the column rather than to a literal, so this also exercises the
	// conversion on a vector rather than on a folded constant.
	{
		query.select("SELECT rowid FROM layers WHERE " + quoted + " = " + conversion + "(" +
		                 conversion + "_text(" + quoted + "))",
		             out);
		PV_VALID(out.bit_count(), row_count);
	}

	return 0;
}
