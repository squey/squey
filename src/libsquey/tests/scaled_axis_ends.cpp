//
// MIT License
//
// © Squey, 2026
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

// Which rows hold the ends of an axis: the parallel view labels its axes with
// their values, so a wrong row is a wrong number printed at the end of an axis.
//
// The rows are found by a parallel walk over the scaled column. Each thread kept a
// running minimum and maximum, and weighed a row against the minimum only when it
// was no new maximum -- which the first row a thread visits always is, unless its
// position is zero. So whenever the only row holding the smallest position was the
// first of a thread's range, it was never counted, and the axis was labelled with
// another row's value. Row 0 starts a range whatever the number of threads.
//
// The smallest position is zero often enough to have hidden this: the default
// scaling places a column's largest value there. A mapping with fixed bounds does
// not -- a time of day mapped to 24h only reaches the end of the axis if something
// happened at the last millisecond of a day -- and neither do the other scalings.

#include "common.h"

#include <squey/PVScaled.h>
#include <squey/PVSelection.h>
#include <squey/PVSource.h>
#include <squey/PVView.h>

#include <pvkernel/core/squey_assert.h>

#include <QTemporaryDir>

#include <omp.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

// Two columns: a time read as seconds since the epoch and mapped to the time of
// day, and a number.
static const std::string fileformat = TEST_FOLDER "/picviz/heat_line.csv.format";

namespace
{

constexpr PVRow ROWS = 1000;
const PVCol TIME(0);
const PVCol NUMBER(1);

/**
 * Check the rows a column names as its ends against a plain walk over it.
 *
 * @param planted the row the data was written to hold the smallest position
 */
void check_ends(const std::string& csv, PVRow planted, const char* what)
{
	pvtest::TestEnv env(csv, fileformat, 1, pvtest::ProcessUntil::View);
	Squey::PVSource& source = *env.root.get_children<Squey::PVSource>().front();
	Squey::PVView& view = *source.current_view();
	const Squey::PVScaled& scaled = view.get_parent<Squey::PVScaled>();
	PV_ASSERT_VALID(scaled.get_row_count() == ROWS, "rows", scaled.get_row_count());

	const uint32_t* const time = scaled.get_column_pointer(TIME);
	const PVRow lowest = PVRow(std::min_element(time, time + ROWS) - time);
	const PVRow highest = PVRow(std::max_element(time, time + ROWS) - time);

	// The data has to set the trap, or passing proves nothing: the planted row holds
	// the only smallest position, and that position is not zero.
	PV_ASSERT_VALID(lowest == planted, "case", what, "smallest position at row", lowest);
	PV_ASSERT_VALID(time[lowest] > 0, "case", what, "smallest position", time[lowest]);
	PV_ASSERT_VALID(std::count(time, time + ROWS, time[lowest]) == 1, "case", what,
	                "rows sharing the smallest position",
	                std::count(time, time + ROWS, time[lowest]));

	const PVRow min_row = scaled.get_col_min_row(TIME);
	const PVRow max_row = scaled.get_col_max_row(TIME);
	std::cout << what << ": the axis ends read "
	          << source.get_rushnraw().at_string(min_row, TIME) << " and "
	          << source.get_rushnraw().at_string(max_row, TIME) << " (rows " << min_row
	          << " and " << max_row << "), the column's are "
	          << source.get_rushnraw().at_string(lowest, TIME) << " and "
	          << source.get_rushnraw().at_string(highest, TIME) << std::endl;
	PV_ASSERT_VALID(min_row == lowest, "case", what, "row named for the smallest position",
	                min_row);
	PV_ASSERT_VALID(max_row == highest, "case", what, "row named for the largest position",
	                max_row);

	// The walk restricted to a selection answers the same over every row.
	Squey::PVSelection all(ROWS);
	all.select_all();
	PVRow sel_min = 0;
	PVRow sel_max = 0;
	scaled.get_col_minmax(sel_min, sel_max, all, TIME);
	PV_ASSERT_VALID(sel_min == lowest and sel_max == highest, "case", what,
	                "selection walk disagrees, min", sel_min);

	// A column holding one value everywhere has every row at both ends, and names
	// the first one for both: the same column has to name the same rows however
	// many threads walked it, and in whichever order they finished.
	PV_ASSERT_VALID(scaled.get_col_min_row(NUMBER) == 0, "case", what,
	                "row named for the smallest of equal positions",
	                scaled.get_col_min_row(NUMBER));
	PV_ASSERT_VALID(scaled.get_col_max_row(NUMBER) == 0, "case", what,
	                "row named for the largest of equal positions",
	                scaled.get_col_max_row(NUMBER));
}

/**
 * A log whose latest time of day is on one row only.
 *
 * Every other row is a whole minute within the first seventeen hours, and the
 * planted one falls at 23:53:20 -- latest in the day, so smallest on the axis, since
 * scaled positions are stored inverted.
 */
std::string write_log(const QTemporaryDir& dir, PVRow planted, const char* name)
{
	const std::string path = dir.filePath(name).toStdString();
	std::ofstream out(path);
	for (PVRow i = 0; i < ROWS; ++i) {
		out << (i == planted ? 86000 : (i + 1) * 60) << ",7\n";
	}
	return path;
}

} // namespace

int main()
{
	QTemporaryDir dir;
	PV_ASSERT_VALID(dir.isValid(), "a temporary directory", "could not be made");

	// Row 0 starts the walk on any machine.
	check_ends(write_log(dir, 0, "first_row.csv"), 0, "first row");

	// Any range start does: with four threads the rows split into four ranges, and
	// the third one starts at row 500.
	omp_set_num_threads(4);
	check_ends(write_log(dir, 500, "range_start.csv"), 500, "start of a range");

	// And a row that starts nothing was always found; kept as the control.
	check_ends(write_log(dir, 321, "inside_a_range.csv"), 321, "inside a range");

	return 0;
}
