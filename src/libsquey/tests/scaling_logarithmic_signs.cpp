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

// A logarithmic axis spreads the magnitudes of the values: every tenfold step takes
// the same height. Negative bounds take the mirror of the positive ones, and bounds
// around 0 keep 0 at 0 and spread the magnitudes of both signs alike, so that a
// column from -1000 to 1000 puts 0 in the middle.

#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVNrawCacheManager.h>
#include <squey/PVScaled.h>
#include <squey/PVSource.h>
#include <squey/PVView.h>

#include "common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

static const char* FORMAT = R"(<?xml version='1.0' encoding='UTF-8'?>
<!DOCTYPE PVParamXml>
<param version="11" first_line="0">
 <splitter type="csv" sep="," quote="&quot;">
  <field>
   <axis name="around_zero" type="number_int64">
    <mapping mode="default"/>
    <scaling mode="log"/>
   </axis>
  </field>
  <field>
   <axis name="negative" type="number_int64">
    <mapping mode="default"/>
    <scaling mode="log"/>
   </axis>
  </field>
  <field>
   <axis name="positive" type="number_int64">
    <mapping mode="default"/>
    <scaling mode="log"/>
   </axis>
  </field>
 </splitter>
</param>
)";

static const char* CSV = "-1000,-10000,1\n"
                         "-10,-1000,10\n"
                         "0,-100,100\n"
                         "10,-10,1000\n"
                         "1000,-1,10000\n";

/**
 * The heights of the rows of column @p col, from 0 for the lowest to 1 for the highest.
 */
static std::vector<double> heights(const Squey::PVScaled& scaled, PVCol col, size_t row_count)
{
	// The positions are stored inverted: the largest one is the bottom of the axis.
	const uint32_t* positions = scaled.get_column_pointer(col);
	std::vector<double> heights;
	for (size_t row = 0; row < row_count; row++) {
		heights.push_back(double(uint32_t(~positions[row])));
	}
	const auto [lowest, highest] = std::minmax_element(heights.begin(), heights.end());
	const double bottom = *lowest;
	const double range = *highest - *lowest;
	for (double& height : heights) {
		height = (height - bottom) / range;
	}
	return heights;
}

static void
check_heights(const Squey::PVScaled& scaled, PVCol col, const std::vector<double>& expected)
{
	const std::vector<double> actual = heights(scaled, col, expected.size());
	for (size_t row = 0; row < expected.size(); row++) {
		PV_ASSERT_VALID(std::abs(actual[row] - expected[row]) < 1e-6, "column", int(col), "row",
		                row, "height", actual[row], "expected", expected[row]);
	}
}

int main()
{
	const std::string dir = PVRush::PVNrawCacheManager::nraw_dir().toStdString();
	const std::string csv = dir + "/logarithmic_signs.csv";
	const std::string format = dir + "/logarithmic_signs.csv.format";
	std::ofstream(csv) << CSV;
	std::ofstream(format) << FORMAT;

	pvtest::TestEnv env(csv, format, 1, pvtest::ProcessUntil::View);
	Squey::PVSource& source = *env.root.get_children<Squey::PVSource>().front();
	PV_VALID((size_t)source.get_rushnraw().row_count(), size_t(5));
	const Squey::PVScaled& scaled = source.current_view()->get_parent<Squey::PVScaled>();

	check_heights(scaled, PVCol(2), {0, .25, .5, .75, 1});
	check_heights(scaled, PVCol(1), {0, .25, .5, .75, 1});

	// The logarithm of one plus the distance to 0, signed, which leaves the values within one
	// of 0 almost evenly spread.
	const double ten = std::log1p(10.) / (2 * std::log1p(1000.));
	check_heights(scaled, PVCol(0), {0, .5 - ten, .5, .5 + ten, 1});

	std::remove(csv.c_str());
	std::remove(format.c_str());
	return 0;
}
