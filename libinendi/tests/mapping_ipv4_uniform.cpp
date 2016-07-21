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

static constexpr const char* csv_file = TEST_FOLDER "/picviz/ipv4_default_mapping.csv";
static constexpr const char* csv_file_format =
    TEST_FOLDER "/picviz/ipv4_default_mapping.csv.format";

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
	pvcop::db::array dist;
	pvcop::db::algo::distinct(column, dist);

	// compute distinct mapping values.
	std::set<uint32_t> distinct_mapping;
	for (size_t i = 0; i < column.size(); i++) {
		distinct_mapping.insert(mapped.get_column(0).to_core_array<uint32_t>()[i]);
	}

	// Check there is a much distinct mapping than distinct values.
	PV_VALID(dist.size(), distinct_mapping.size());

	// Check it is equi-reparteed
	PV_ASSERT_VALID(std::adjacent_find(distinct_mapping.begin(), distinct_mapping.end(),
	                                   [](uint32_t a, uint32_t b) {
		                                   return b + 1 < a or a < b - 1;
		                               }) != distinct_mapping.end());
#else
	(void)mapped;
#endif

	return 0;
}
