/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVMapped.h>
#include <pvkernel/core/inendi_assert.h>
#include <pvcop/db/algo.h>

#include "common.h"

#include <chrono>
#include <iostream>

#ifdef INSPECTOR_BENCH
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

	Inendi::PVMapped& mapped = env.compute_mapping();

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;

	std::cout << diff.count();

#ifndef INSPECTOR_BENCH
	// Compute distinct values.
	PVRush::PVNraw const& nraw = env.root.get_children<Inendi::PVSource>().front()->get_rushnraw();
	const pvcop::db::array& column = nraw.collection().column(0);

	for (size_t i = 0; i < column.size() / 2; i++) {

		// Check IP are in 0 -> 2*31
		uint32_t map_ip = mapped.get_column(0).to_core_array<uint32_t>()[i];
		PV_ASSERT_VALID(map_ip < (1UL << 31));

		// Check str are in 2*31 -> 2**32
		uint32_t map_str = mapped.get_column(1).to_core_array<uint32_t>()[i];
		PV_ASSERT_VALID(map_str >= (1UL << 31));
	}
#endif

	return 0;
}
