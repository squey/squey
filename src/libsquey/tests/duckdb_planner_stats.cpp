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

// A scope is a selection, so how many rows it holds is a count of set bits
// rather than a guess. DuckDB is told, because a planner with nothing to go on
// plans for the worst: which side of a join to build a hash table on, and how
// much room to set aside for it, are decided from that number.
//
// The plan is where it shows, which is why this reads one. Nothing else does:
// a better plan and a worse plan give the same answer, so a test checking the
// result would pass either way.

#include <squey/PVDuckDBQuery.h>
#include <squey/PVSelection.h>
#include <squey/PVView.h>

#include <pvkernel/core/squey_assert.h>

#include <string>

#include "common.h"

const std::string filename = TEST_FOLDER "/picviz/errors_search.csv";
const std::string fileformat = TEST_FOLDER "/picviz/errors_search.csv.format";

/**
 * The plan DuckDB drew for @a sql, as one string.
 */
static std::string plan_of(const Squey::PVDuckDBQuery& query, const std::string& sql)
{
	std::string drawn;
	for (const auto& row : query.run_tabular("EXPLAIN " + sql).rows) {
		for (const auto& cell : row) {
			drawn += cell;
		}
	}
	return drawn;
}

int main()
{
	pvtest::TestEnv env(filename, fileformat, 1, pvtest::ProcessUntil::View);
	Squey::PVView* view = env.root.current_view();
	Squey::PVDuckDBQuery query(*view);
	const size_t row_count = view->get_row_count();
	PV_ASSERT_VALID(row_count > 3, "the file is too small to tell the scopes apart", row_count);

	// --- Every row, which is what the stack lets through here ------------------
	{
		const std::string plan = plan_of(query, "SELECT rowid FROM layers");
		PV_ASSERT_VALID(plan.find("~" + std::to_string(row_count) + " rows") != std::string::npos,
		                "the plan does not expect the rows the scope holds", plan);
	}

	// --- And a selection of three, under the same query object -----------------
	// Resolved when the query is planned rather than when the object was made,
	// so this is the number the scan is about to read.
	{
		Squey::PVSelection three(row_count);
		three.select_none();
		for (PVRow row = 0; row < 3; ++row) {
			three.set_line(row, true);
		}
		view->set_selection_view(three);

		const std::string plan = plan_of(query, "SELECT rowid FROM selection");
		PV_ASSERT_VALID(plan.find("~3 rows") != std::string::npos,
		                "the plan does not expect what the selection holds", plan);
		// And not the whole source, which is what it would say knowing nothing.
		PV_ASSERT_VALID(plan.find("~" + std::to_string(row_count) + " rows") == std::string::npos,
		                "the plan expects every row of the source", plan);
	}

	return 0;
}
