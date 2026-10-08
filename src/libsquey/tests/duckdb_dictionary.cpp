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

// A string column reaches SQL by one of two routes, and this checks they answer
// the same thing.
//
// pvcop stores a string column as one dictionary index per row plus the distinct
// strings once. The scan can hand that shape straight to DuckDB as a dictionary
// vector -- one string built per distinct value, per query -- or build the text
// of every row it reads. Which one is cheaper depends on how many rows the query
// reads against how many distinct values the column holds, so the scan picks per
// query, and both routes have to be indistinguishable from the outside.
//
// The two cases below sit on either side of that threshold, and each asserts the
// arithmetic that puts it there rather than assuming it: a query over the whole
// column reads more rows than the dictionary has entries, one restricted to a
// handful of rows reads far fewer.

#include <squey/PVDuckDBQuery.h>
#include <squey/PVSelection.h>
#include <squey/PVSource.h>
#include <squey/PVView.h>

#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVNraw.h>

#include <pvcop/db/array.h>

#include <string>

#include "common.h"

// 50k rows over ~40k distinct values: the dictionary is smaller than the column,
// which is what a query over the whole source needs for the dictionary route.
const std::string filename = TEST_FOLDER "/picviz/string_mapping.csv";
const std::string fileformat = TEST_FOLDER "/picviz/string_mapping.csv.format";

constexpr PVCol STRING_COL(0);
constexpr char STRING_COL_NAME[] = "string_1";

int main()
{
	pvtest::TestEnv env(filename, fileformat, 1, pvtest::ProcessUntil::View);
	Squey::PVView* view = env.root.current_view();
	const PVRush::PVNraw& nraw = view->get_rushnraw_parent();
	const size_t row_count = nraw.row_count();

	Squey::PVDuckDBQuery query(view->get_parent<Squey::PVSource>());

	const std::string col = Squey::PVDuckDBQuery::quote_identifier(STRING_COL_NAME);
	PV_VALID(query.column_names()[STRING_COL + 1], std::string(STRING_COL_NAME));
	PV_VALID(query.column_types()[STRING_COL + 1], std::string("VARCHAR"));

	// The number of distinct values is what the dictionary holds, and comparing
	// it against the rows a query reads is what picks the route.
	const auto distinct_table =
	    query.run_tabular("SELECT COUNT(DISTINCT " + col + ") FROM layers");
	const size_t distinct = std::stoull(distinct_table.rows[0][0]);

	// --- Whole column: more rows read than the dictionary holds ---------------
	{
		PV_ASSERT_VALID(distinct < row_count, "distinct", distinct, "rows", row_count);

		const auto table =
		    query.run_tabular("SELECT rowid, " + col + " FROM layers", nullptr, row_count);
		PV_VALID(table.rows.size(), row_count);
		PV_ASSERT_VALID(not table.truncated, "truncated", 1);

		// pvcop is the oracle: whichever route the scan took, the text of a row
		// has to be the text pvcop gives for it.
		const pvcop::db::array& column = nraw.column(STRING_COL);
		for (const auto& row : table.rows) {
			const size_t rowid = std::stoull(row[0]);
			PV_ASSERT_VALID(row[1] == column.at(rowid), "rowid", rowid, "sql", row[1], "pvcop",
			                column.at(rowid));
		}
	}

	// --- A handful of rows: fewer read than the dictionary holds --------------
	{
		Squey::PVSelection in(row_count);
		in.select_none();
		std::vector<size_t> picked;
		for (size_t row = 0; row < row_count; row += row_count / 10) {
			in.set_line(PVRow(row), true);
			picked.push_back(row);
		}
		PV_ASSERT_VALID(picked.size() < distinct, "picked", picked.size(), "distinct", distinct);

		const auto table =
		    query.run_tabular("SELECT rowid, " + col + " FROM selection ORDER BY rowid", &in, row_count);
		PV_VALID(table.rows.size(), picked.size());

		const pvcop::db::array& column = nraw.column(STRING_COL);
		for (size_t i = 0; i < table.rows.size(); ++i) {
			PV_VALID(size_t(std::stoull(table.rows[i][0])), picked[i]);
			PV_ASSERT_VALID(table.rows[i][1] == column.at(picked[i]), "rowid", picked[i], "sql",
			                table.rows[i][1], "pvcop", column.at(picked[i]));
		}
	}

	// --- A predicate must select the same rows either way ---------------------
	// The route changes what the scan hands to DuckDB, so a comparison against a
	// literal is where a mismatched dictionary index would surface as a wrong
	// selection rather than as wrong text.
	{
		const pvcop::db::array& column = nraw.column(STRING_COL);
		const std::string value = column.at(0);
		const std::string predicate = col + " = '" + value + "'";

		// Whole column: the dictionary route.
		size_t expected_all = 0;
		for (size_t row = 0; row < row_count; ++row) {
			expected_all += static_cast<size_t>(column.at(row) == value);
		}
		PV_ASSERT_VALID(expected_all > 0, "expected_all", expected_all);

		Squey::PVSelection whole(row_count);
		query.select(predicate, whole);
		PV_VALID(whole.bit_count(), expected_all);

		// A window narrower than the dictionary: the other route, and the rows it
		// finds have to be exactly those the first one found inside that window.
		constexpr size_t WINDOW = 1000;
		PV_ASSERT_VALID(WINDOW < distinct, "window", WINDOW, "distinct", distinct);

		Squey::PVSelection in(row_count);
		in.select_none();
		for (size_t row = 0; row < WINDOW; ++row) {
			in.set_line(PVRow(row), true);
		}

		Squey::PVSelection restricted(row_count);
		query.select(predicate, in, restricted);

		Squey::PVSelection expected_window = whole;
		expected_window &= in;
		PV_ASSERT_VALID((restricted ^ expected_window).is_empty(), "restricted",
		                restricted.bit_count(), "expected", expected_window.bit_count());
		PV_ASSERT_VALID(restricted.bit_count() > 0, "restricted", restricted.bit_count());
	}

	return 0;
}
