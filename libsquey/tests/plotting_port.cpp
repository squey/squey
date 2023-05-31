//
// MIT License
//
// © ESI Group, 2015
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

#include <squey/PVMapped.h>
#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVUtils.h>

#include "common.h"

#include <chrono>
#include <iostream>

#ifdef SQUEY_BENCH
// 10 000 000 lines.
static constexpr int dupl = 200;
#else
static constexpr int dupl = 1;
#endif
using plotting_t = Squey::PVPlottingFilter::value_type;
using port_plotting_t = uint16_t;

static constexpr const char* csv_file = TEST_FOLDER "/picviz/plotting_port.csv";
static constexpr const char* csv_file_format = TEST_FOLDER "/picviz/plotting_port.csv.format";
static constexpr const plotting_t threshold1 = (0.3 * std::numeric_limits<plotting_t>::max()) - 1;
static constexpr const plotting_t threshold2 = (0.6 * std::numeric_limits<plotting_t>::max()) - 1;
static constexpr const plotting_t max_threshold = std::numeric_limits<plotting_t>::max();

int main()
{
	pvtest::TestEnv env(csv_file, csv_file_format, dupl, pvtest::ProcessUntil::Mapped);

	auto start = std::chrono::system_clock::now();

	Squey::PVPlotted const& plotted = env.compute_plotting();

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;

	std::cout << diff.count();

#ifndef SQUEY_BENCH
	// Compute distinct values.
	PVRush::PVNraw const& nraw = env.root.get_children<Squey::PVSource>().front()->get_rushnraw();
	const pvcop::db::array& column = nraw.column(PVCol(0));
	auto& array = column.to_core_array<port_plotting_t>();

	for (size_t i = 1; i < array.size(); i++) {
		PV_ASSERT_VALID(plotted.get_column_pointer(PVCol(0))[i] <
		                plotted.get_column_pointer(PVCol(0))[i - 1]);
		if (i < 1024) {
			PV_ASSERT_VALID(~plotting_t(plotted.get_column_pointer(PVCol(0))[i]) <= threshold1);
		} else if (i >= 1024 && i <= 49151) {
			PV_ASSERT_VALID(~plotting_t(plotted.get_column_pointer(PVCol(0))[i]) > threshold1 &&
			                ~plotting_t(plotted.get_column_pointer(PVCol(0))[i]) <= threshold2);
		} else {

			PV_ASSERT_VALID(~plotting_t(plotted.get_column_pointer(PVCol(0))[i]) > threshold2 &&
			                ~plotting_t(plotted.get_column_pointer(PVCol(0))[i]) <= max_threshold);
		}
	}

#else
	(void)plotted;
	(void)threshold1;
	(void)threshold2;
	(void)max_threshold;
#endif

	return 0;
}
