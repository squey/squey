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

/**
 * reference mapping result are contained in the input CSV file with contains:
 * - col(0): an index
 * - col(1): a MAC address
 * - col(2): reference linear mapping value
 * - col(3): reference uniform mapping value
 * - col(4): reference linear/uniform mapping value
 * - col(5): reference uniform/linear mapping value
 * - col(6): reference uniform/uniform mapping value
 */
static constexpr const char* csv_file = TEST_FOLDER "/picviz/mac_address.csv";

int main(int argc, char** argv)
{
	if (argc != 3) {
		std::cerr << "usage: " << basename(argv[0]) << " ref_col file.csv" << std::endl;
		return 1;
	}

	PVCol ref_col = (PVCol)atoi(argv[1]);
	const char* format_file = argv[2];

	pvtest::TestEnv env(csv_file, format_file, 1);

	auto start = std::chrono::system_clock::now();

	Squey::PVMapped& mapped = env.compute_mapping();

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;

	std::cout << diff.count();

	// Check computed mapping is equal to the reference value
	PVRush::PVNraw const& nraw = env.root.get_children<Squey::PVSource>().front()->get_rushnraw();
	const pvcop::db::array& ref_array = nraw.column(ref_col);
	const auto& ref_values = ref_array.to_core_array<uint32_t>();
	const auto& mapping = mapped.get_column(PVCol(1)).to_core_array<uint32_t>();

	PV_VALID(mapping.size(), ref_values.size());

	for (size_t i = 0; i < ref_values.size(); i++) {
		PV_VALID(mapping[i], ref_values[i], "i", i);
	}

	return 0;
}
