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

#include <squey/PVPlotted.h>
#include <pvkernel/core/squey_assert.h>
#include <pvcop/db/algo.h>

#include "common.h"

#include <chrono>
#include <iostream>

#ifdef SQUEY_BENCH
// 10 000 000 lines.
static constexpr int dupl = 200;
#else
static constexpr int dupl = 1;
#endif

static constexpr const char* csv_file = TEST_FOLDER "/picviz/enum_mapping.csv";
static constexpr const char* csv_file_format = TEST_FOLDER "/picviz/enum_mapping.csv.format";

int main()
{
	pvtest::TestEnv env(csv_file, csv_file_format, dupl, pvtest::ProcessUntil::Plotted);

	auto start = std::chrono::system_clock::now();

	Squey::PVPlotted const& plotted = env.compute_plotting();

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;

	std::cout << diff.count();

#ifndef SQUEY_BENCH
	// Compute distinct values.
	PVRush::PVNraw const& nraw = env.root.get_children<Squey::PVSource>().front()->get_rushnraw();
	const pvcop::db::array& column = nraw.column(PVCol(0));
	pvcop::db::array dist;
	pvcop::db::algo::distinct(column, dist);

	// compute distinct plotting values.
	std::set<uint32_t> distinct_plotting;
	for (size_t i = 0; i < column.size(); i++) {
		distinct_plotting.insert(plotted.get_column_pointer(PVCol(0))[i]);
	}

	auto minmax = std::minmax_element(distinct_plotting.begin(), distinct_plotting.end());

	PV_VALID(*minmax.first, 0U);
	PV_VALID(*minmax.second, std::numeric_limits<uint32_t>::max());

	// Check there is a much distinct plotting than distinct values.
	PV_VALID(dist.size(), distinct_plotting.size());

	// Check it is equi-reparteed
	PV_ASSERT_VALID(std::adjacent_find(distinct_plotting.begin(), distinct_plotting.end(),
	                                   [](uint32_t a, uint32_t b) {
		                                   return b + 1 < a or a < b - 1;
		                               }) != distinct_plotting.end());
#else
	(void)plotted;
#endif

	return 0;
}
