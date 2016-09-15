/**
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2016
 */

#include <inendi/PVMapped.h>
#include <pvkernel/core/inendi_assert.h>
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

	int ref_col = atoi(argv[1]);
	const char* format_file = argv[2];

	pvtest::TestEnv env(csv_file, format_file, 1);

	auto start = std::chrono::system_clock::now();

	Inendi::PVMapped& mapped = env.compute_mapping();

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;

	std::cout << diff.count();

	// Check computed mapping is equal to the reference value
	PVRush::PVNraw const& nraw = env.root.get_children<Inendi::PVSource>().front()->get_rushnraw();
	const auto ref_array = nraw.collection().column(ref_col);
	const auto& ref_values = ref_array.to_core_array<uint32_t>();
	const auto& mapping = mapped.get_column(1).to_core_array<uint32_t>();

	PV_VALID(mapping.size(), ref_values.size());

	for (size_t i = 0; i < ref_values.size(); i++) {
		PV_VALID(mapping[i], ref_values[i], "i", i);
	}

	return 0;
}
