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

static constexpr const char* csv_file = TEST_FOLDER "/picviz/host_default_mapping.csv";
static constexpr const char* csv_file_format =
    TEST_FOLDER "/picviz/host_default_mapping.csv.format";

int main()
{
	pvtest::TestEnv env(csv_file, csv_file_format, dupl);

	auto start = std::chrono::system_clock::now();

	Squey::PVMapped& mapped = env.compute_mapping();

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;

	std::cout << diff.count();

#ifndef SQUEY_BENCH
	// Compute distinct values.
	PVRush::PVNraw const& nraw = env.root.get_children<Squey::PVSource>().front()->get_rushnraw();
	const pvcop::db::array& column = nraw.column(PVCol(0));

	for (size_t i = 0; i < column.size() / 2; i++) {

		// Check IP are in 0 -> 2*31
		uint32_t map_ip = mapped.get_column(PVCol(0)).to_core_array<uint32_t>()[i];
		PV_ASSERT_VALID(map_ip < (1UL << 31));

		// Check str are in 2*31 -> 2**32
		uint32_t map_str = mapped.get_column(PVCol(1)).to_core_array<uint32_t>()[i];
		PV_ASSERT_VALID(map_str >= (1UL << 31));
	}
#else
	(void)mapped;
#endif

	return 0;
}
