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

// Checks that a SQL query over the pvcop columns produces the same selection as
// reading the column directly. The oracle is always pvcop itself: the point of
// the scan is that DuckDB sees exactly what Squey sees.
//
// The sparse cases matter most. A block of rows holding no selected line must
// not end the scan -- returning an empty chunk is how DuckDB is told a thread
// is done -- so a selection with large empty stretches would silently truncate
// the result if the scan did not keep claiming blocks.

#include <squey/PVDuckDBQuery.h>
#include <squey/PVSource.h>
#include <squey/PVView.h>

#include <pvkernel/core/PVSelBitField.h>
#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVNraw.h>

#include <pvcop/db/array.h>

#include <string>
#include <vector>

#include "common.h"

const std::string filename = TEST_FOLDER "/picviz/heat_line.csv";
const std::string fileformat = TEST_FOLDER "/picviz/heat_line.csv.format";

// The second axis of the test file holds one unsigned integer per row, which is
// what makes the zero-copy path exercised rather than the textual fallback.
constexpr PVCol VALUE_COL(1);

/**
 * Count the selected rows explicitly rather than through bit_count(), whose
 * default end bound makes the last row ambiguous.
 */
static size_t count_selected(const PVCore::PVSelBitField& sel, size_t row_count)
{
	size_t count = 0;
	for (size_t row = 0; row < row_count; ++row) {
		count += static_cast<size_t>(sel.get_line(PVRow(row)));
	}
	return count;
}

/**
 * Count the rows whose uint8 column is below @a threshold, reading pvcop
 * directly. This is the oracle the SQL result is compared against.
 */
static size_t count_below(const PVRush::PVNraw& nraw, uint64_t threshold)
{
	const pvcop::db::array& column = nraw.column(VALUE_COL);
	size_t count = 0;
	for (size_t row = 0; row < column.size(); ++row) {
		if (std::stoull(column.at(row)) < threshold) {
			++count;
		}
	}
	return count;
}

int main()
{
	pvtest::TestEnv env(filename, fileformat, 1, pvtest::ProcessUntil::View);
	Squey::PVView* view = env.root.current_view();
	const PVRush::PVNraw& nraw = view->get_rushnraw_parent();
	const size_t row_count = nraw.row_count();

	PV_ASSERT_VALID(row_count > 0, "row_count", row_count);

	// Built from the source, so the column names come from the format rather
	// than from the caller.
	Squey::PVDuckDBQuery query(view->get_parent<Squey::PVSource>());

	// The schema must expose rowid plus one name per nraw column.
	const std::vector<std::string> names = query.column_names();
	PV_VALID(names.size(), size_t(nraw.column_count()) + 1);
	PV_VALID(names[0], std::string("rowid"));
	PV_VALID(names[2], std::string("uint8"));

	// --- Identifier quoting ---------------------------------------------------
	// Axis names are exposed verbatim rather than folded into an SQL-safe shape,
	// so anything that is not a plain identifier has to be quoted to be written
	// into a query. The dashed name below is taken from a real pcap format.
	{
		using Q = Squey::PVDuckDBQuery;
		PV_VALID(Q::quote_identifier("uint8"), std::string("uint8"));
		PV_VALID(Q::quote_identifier("arp.duplicate-address-detected"),
		         std::string("\"arp.duplicate-address-detected\""));
		PV_VALID(Q::quote_identifier("Axis 1"), std::string("\"Axis 1\""));
		PV_VALID(Q::quote_identifier("2nd"), std::string("\"2nd\""));
		// A quote inside the name is escaped by doubling it.
		PV_VALID(Q::quote_identifier("a\"b"), std::string("\"a\"\"b\""));
	}

	PVCore::PVSelBitField out(row_count);

	// --- Whole source, dense regime ------------------------------------------
	{
		const uint64_t threshold = 12499;
		query.select("SELECT rowid FROM layers WHERE uint8 < " + std::to_string(threshold),
		             out);
		PV_VALID(count_selected(out, row_count), count_below(nraw, threshold));
	}

	// --- Bare predicate ------------------------------------------------------
	// The condition alone must select the same rows as the statement spelling
	// it out: only the WHERE clause carries intent, so that is the form users
	// are given.
	{
		const uint64_t threshold = 12499;
		PVCore::PVSelBitField predicate_out(row_count);
		query.select("uint8 < " + std::to_string(threshold), predicate_out);
		PV_VALID(count_selected(predicate_out, row_count), count_below(nraw, threshold));

		// Wrapping parenthesizes the predicate, so an OR cannot leak out of it
		// and widen the selection beyond what was asked.
		//
		// This one carries a second job. DuckDB pushes it as an optional filter,
		// which pvcop declines -- an OR of inequalities is not membership -- so
		// the scan evaluates it on the emitted chunk rather than trusting an
		// operator above to do it. The count below is what says that happened.
		// Do not weaken it into a range check.
		query.select("uint8 < 10 OR uint8 > 40000", out);
		size_t expected = 0;
		const pvcop::db::array& column = nraw.column(VALUE_COL);
		for (size_t row = 0; row < row_count; ++row) {
			const uint64_t v = std::stoull(column.at(row));
			expected += static_cast<size_t>(v < 10 || v > 40000);
		}
		PV_VALID(count_selected(out, row_count), expected);
		// Nothing was let go: an OR of inequalities has a static form, so the
		// scan applies it. Only a filter DuckDB renders as the constant true --
		// one whose value moves as the query runs -- is dropped.
		PV_VALID(query.dropped_optional_filters(), size_t(0));

		// And the case the scan really does let go, which has to stay exercised or
		// the rule that recognises it would rot: "ORDER BY … LIMIT" pushes a
		// filter whose value moves as the query runs. It has no static form, and
		// DuckDB says so by rendering it as the constant true rather than by
		// refusing -- which is what the scan reads to tell a predicate it owes an
		// answer to from a hint it may ignore.
		query.run_tabular("SELECT rowid FROM layers ORDER BY uint8 LIMIT 5", nullptr, 5);
		PV_ASSERT_VALID(query.dropped_optional_filters() > 0, "nothing was let go",
		                query.dropped_optional_filters());

		// A statement is still recognised as such and passed through.
		query.select("  \n SELECT rowid FROM layers WHERE uint8 < 10", out);
		size_t below_ten = 0;
		for (size_t row = 0; row < row_count; ++row) {
			below_ten += static_cast<size_t>(std::stoull(column.at(row)) < 10);
		}
		PV_VALID(count_selected(out, row_count), below_ten);
	}

	// Every row matches: the scan stays on the contiguous path from end to end.
	{
		query.select("SELECT rowid FROM layers", out);
		PV_VALID(count_selected(out, row_count), row_count);
	}

	// No row matches: the result is empty without the scan stalling.
	{
		query.select("SELECT rowid FROM layers WHERE uint8 < 0", out);
		PV_VALID(count_selected(out, row_count), size_t(0));
	}

	// --- Restricted to an input selection, sparse regime ----------------------
	{
		// One row in a thousand, so most 256k-row blocks still hold a few rows
		// while the bitfield is mostly empty.
		PVCore::PVSelBitField in(row_count);
		in.select_none();
		size_t expected = 0;
		for (size_t row = 0; row < row_count; row += 1000) {
			in.set_line(PVRow(row), true);
			++expected;
		}

		query.select("SELECT rowid FROM selection", in, out);
		PV_VALID(count_selected(out, row_count), expected);

		// The rows returned must be exactly those of the input selection.
		for (size_t row = 0; row < row_count; ++row) {
			PV_ASSERT_VALID(out.get_line(PVRow(row)) == in.get_line(PVRow(row)), "row", row);
		}
	}

	// A selection confined to the very end of the source: every block before it
	// is empty, which is the case that truncates a scan that treats an empty
	// block as the end of its work.
	{
		PVCore::PVSelBitField in(row_count);
		in.select_none();
		const size_t first = row_count - 1;
		in.set_line(PVRow(first), true);

		query.select("SELECT rowid FROM selection", in, out);
		PV_VALID(count_selected(out, row_count), size_t(1));
		PV_ASSERT_VALID(out.get_line(PVRow(first)), "first", first);
	}

	// An empty input selection yields nothing, and a predicate over a restricted
	// selection composes the two the way an input selection does in pvcop.
	{
		PVCore::PVSelBitField in(row_count);
		in.select_none();
		query.select("SELECT rowid FROM selection", in, out);
		PV_VALID(count_selected(out, row_count), size_t(0));

		in.select_all();
		const uint64_t threshold = 12499;
		query.select("SELECT rowid FROM selection WHERE uint8 < " + std::to_string(threshold), in,
		             out);
		PV_VALID(count_selected(out, row_count), count_below(nraw, threshold));
	}

	// --- Results that are not selections --------------------------------------
	// An aggregate returns rows that are not rows of the source, so it cannot
	// become a selection -- but it is what one writes to understand a dataset,
	// so it must be retrievable as a table rather than rejected.
	{
		PV_ASSERT_VALID(query.yields_selection("SELECT rowid FROM layers"), "yields", 1);
		PV_ASSERT_VALID(query.yields_selection("uint8 < 10"), "predicate yields", 1);
		PV_ASSERT_VALID(not query.yields_selection("SELECT COUNT(*) FROM layers"), "count", 0);
		PV_ASSERT_VALID(not query.yields_selection("SELECT uint8, COUNT(*) FROM layers GROUP BY uint8"),
		                "group by", 0);
		// A malformed query yields no selection either, rather than throwing
		// here: the error belongs to the run, not to the classification.
		PV_ASSERT_VALID(not query.yields_selection("SELECT FROM WHERE"), "malformed", 0);

		const auto table = query.run_tabular("SELECT COUNT(*) AS n FROM layers");
		PV_VALID(table.column_names.size(), size_t(1));
		PV_VALID(table.column_names[0], std::string("n"));
		PV_VALID(table.rows.size(), size_t(1));
		PV_VALID(table.rows[0][0], std::to_string(row_count));
		PV_ASSERT_VALID(not table.truncated, "truncated", 0);

		// The cap is what keeps a large result from being materialized whole.
		const auto capped = query.run_tabular("SELECT rowid, uint8 FROM layers", nullptr, 10);
		PV_VALID(capped.rows.size(), size_t(10));
		PV_VALID(capped.column_names.size(), size_t(2));
		PV_ASSERT_VALID(capped.truncated, "truncated", 1);
	}

	// --- A subquery over the same table ---------------------------------------
	// "the rows whose group holds a single value of the other column" is the
	// shape a console question takes, and it reads the scan twice. DuckDB turns
	// the IN into a semi-join and pushes a filter of its own into the outer scan
	// -- and a filter this scan accepts is one the operator above it no longer
	// applies, so getting that wrong would widen the result silently.
	//
	// The oracle is the same predicate with the inner result written out as a
	// literal list: same rows, without the pushed filter.
	{
		const std::string inner =
		    "SELECT uint8 FROM layers GROUP BY 1 HAVING COUNT(DISTINCT datetime) = 1";

		// Uncapped: the default cap would silently shorten the list and make the
		// comparison pass against a subset of itself.
		const auto groups = query.run_tabular(inner, nullptr, row_count);
		PV_ASSERT_VALID(not groups.rows.empty(), "the file must hold such a group", 0);
		PV_ASSERT_VALID(not groups.truncated, "the oracle list was cut short", groups.rows.size());

		std::string list;
		for (const auto& group : groups.rows) {
			list += (list.empty() ? "" : ", ") + group[0];
		}

		Squey::PVSelection from_subquery(row_count);
		query.select("SELECT rowid FROM layers WHERE uint8 IN (" + inner + ")", from_subquery);
		Squey::PVSelection from_list(row_count);
		query.select("SELECT rowid FROM layers WHERE uint8 IN (" + list + ")", from_list);

		const size_t selected = count_selected(from_subquery, row_count);
		PV_ASSERT_VALID(selected > 0 && selected < row_count, "a vacuous comparison", selected);
		PV_ASSERT_VALID((from_subquery ^ from_list).is_empty(), "the subquery selected other rows",
		                selected);
	}

	// --- Malformed queries must be reported, not silently mis-read ------------
	{
		bool threw = false;
		try {
			query.select("SELECT rowid, uint8 FROM layers", out);
		} catch (const std::runtime_error&) {
			threw = true;
		}
		PV_ASSERT_VALID(threw, "rejected", std::string("two projected columns"));

		threw = false;
		try {
			query.select("SELECT * FROM no_such_table", out);
		} catch (const std::runtime_error&) {
			threw = true;
		}
		PV_ASSERT_VALID(threw, "reported", std::string("unknown table"));

		// Double-quoting a string literal names a column instead, and the raw
		// error says nothing about quoting. The hint is what turns it into
		// something actionable, so it is part of the contract.
		std::string message;
		try {
			query.select("SELECT rowid FROM layers WHERE uint8 = \"some_value\"", out);
		} catch (const std::runtime_error& e) {
			message = e.what();
		}
		PV_ASSERT_VALID(message.find("single quotes") != std::string::npos, "hint", message);
	}

	return 0;
}
