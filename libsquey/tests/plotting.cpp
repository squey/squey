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

int main(int argc, char** argv)
{
	if (argc < 3) {
		std::cerr << "Usage: " << argv[0] << " input_file format" << std::endl;
		return 1;
	}

	const char* input_file = argv[1];
	const char* format = argv[2];

	pvtest::TestEnv env(input_file, format, dupl, pvtest::ProcessUntil::Mapped);

	auto start = std::chrono::system_clock::now();

	Squey::PVPlotted const& plotted = env.compute_plotting();

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;

	std::cout << diff.count();

#ifndef SQUEY_BENCH
	// Check mapping is the same as NRaw value.
	PVRush::PVNraw const& nraw = env.root.get_children<Squey::PVSource>().front()->get_rushnraw();
	const pvcop::db::array& column = nraw.column(PVCol(0));

	pvcop::db::indexes indexes = column.parallel_sort();
	auto& order = indexes.to_core_array();
	std::reverse(order.begin(), order.end()); // plotteds are still inverted...

	PV_VALID(plotted.get_column_pointer(PVCol(0))[order[order.size() - 1]],
	         std::numeric_limits<uint32_t>::max());
	PV_ASSERT_VALID(plotted.get_column_pointer(PVCol(0))[order[0]] <= (uint32_t)1);

	// Check we keep value ordering.
	for (size_t i = 1; i < column.size(); i++) {
		PV_ASSERT_VALID(plotted.get_column_pointer(PVCol(0))[order[i - 1]] <=
		                plotted.get_column_pointer(PVCol(0))[order[i]]);
	}

#else
	(void)plotted;
#endif

	return 0;
}
