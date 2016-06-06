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

	Inendi::PVMapped_p mapped = env.compute_mapping();

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;

	std::cout << diff.count();

#ifndef INSPECTOR_BENCH
	// Compute distinct values.
	auto const& scene = env.root->get_children().front();
	PVRush::PVNraw const& nraw = scene->get_children().front()->get_rushnraw();
	const pvcop::db::array& column = nraw.collection().column(0);

	for (size_t i = 0; i < column.size() / 2; i++) {

		// Check IP are in 0 -> 2*31
		uint32_t map_ip = mapped->get_value(i, 0).storage_as_uint();
		PV_ASSERT_VALID(map_ip < (1UL << 31));

		// Check str are in 2*31 -> 2**32
		uint32_t map_str = mapped->get_value(i, 1).storage_as_uint();
		PV_ASSERT_VALID(map_str >= (1UL << 31));
	}
#endif

	return 0;
}
