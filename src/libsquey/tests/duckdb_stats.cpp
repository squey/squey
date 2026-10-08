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

// Checks the data a GROUP BY hands to the distinct-values listing: the shape
// that makes it recognisable as a value/count pair, and counts that actually
// carry a number. The listing draws its second column through a delegate rather
// than as text, so an empty count there is indistinguishable from a count the
// query never produced -- this is what tells the two apart.
//
// What the dialog then does with that data is checked in libpvguiqt
// (Tqt_sql_console_result), which is where the widgets live.

#include <squey/PVDuckDBQuery.h>
#include <squey/PVSource.h>
#include <squey/PVView.h>

#include <pvkernel/core/squey_assert.h>

#include <string>

#include "common.h"

int main()
{
	pvtest::TestEnv env(TEST_FOLDER "/picviz/heat_line.csv",
	                    TEST_FOLDER "/picviz/heat_line.csv.format", 1,
	                    pvtest::ProcessUntil::View);
	Squey::PVView* view = env.root.current_view();

	Squey::PVDuckDBQuery query(view->get_parent<Squey::PVSource>());
	const auto table = query.run_tabular(
	    "SELECT uint8 AS v, COUNT(*) AS n FROM layers GROUP BY 1 ORDER BY n DESC LIMIT 20");

	// Two columns whose second one is integral: that is what routes the result
	// to the distinct-values listing rather than to a plain grid.
	PV_ASSERT_VALID(table.is_value_count(), "is_value_count", int(table.is_value_count()));
	PV_VALID(table.column_names.size(), size_t(2));
	PV_ASSERT_VALID(not table.rows.empty(), "rows", table.rows.size());

	// A group holds at least the row it was formed from, so a count that is
	// empty or zero means the column was never filled.
	for (const auto& row : table.rows) {
		PV_ASSERT_VALID(not row[1].empty(), "count empty at", row[0]);
		PV_ASSERT_VALID(std::stoull(row[1]) > 0, "count zero at", row[0]);
	}

	return 0;
}
