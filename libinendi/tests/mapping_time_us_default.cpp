/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVMapped.h>
#include <pvkernel/core/inendi_assert.h>
#include <pvkernel/rush/PVUtils.h>

#include "common.h"

#include <chrono>
#include <iostream>

#ifdef INSPECTOR_BENCH
// 10 000 000 lines.
static constexpr int dupl = 200;
#else
static constexpr int dupl = 1;
#endif

static constexpr const char* csv_file = TEST_FOLDER "/picviz/time_mapping_us.csv";
static constexpr const char* csv_file_format =
    TEST_FOLDER "/picviz/datetime_us_default_mapping.csv.format";

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
	const pvcop::db::array& column = nraw.column(PVCol(0));
	auto& array = column.to_core_array<uint64_t>();

	std::vector<uint64_t> order(column.size());
	std::iota(order.begin(), order.end(), 0);

	std::sort(order.begin(), order.end(),
	          [&array](uint64_t a, uint64_t b) { return array[a] < array[b]; });

	uint64_t prev = mapped.get_column(PVCol(0)).to_core_array<uint64_t>()[order[0]];
	for (size_t i = 0; i < column.size(); i++) {
		PV_ASSERT_VALID(prev <= mapped.get_column(PVCol(0)).to_core_array<uint64_t>()[order[i]]);
		prev = mapped.get_column(PVCol(0)).to_core_array<uint64_t>()[order[i]];
	}
#else
	(void)mapped;
#endif

	return 0;
}
