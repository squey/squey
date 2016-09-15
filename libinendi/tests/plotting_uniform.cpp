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

int main(int argc, char** argv)
{
	if (argc < 3) {
		std::cerr << "Usage: " << argv[0] << " input_csv_file format" << std::endl;
		return 1;
	}

	const char* csv_file = argv[1];
	const char* csv_file_format = argv[2];

	pvtest::TestEnv env(csv_file, csv_file_format, dupl, pvtest::ProcessUntil::Plotted);

	auto start = std::chrono::system_clock::now();

	const Inendi::PVPlotted& plotted = env.compute_plotting();

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
	const uint32_t* plotting = plotted.get_column_pointer(0);
	std::set<uint32_t> distinct_plotting;
	for (size_t i = 0; i < column.size(); i++) {
		distinct_plotting.insert(plotting[i]);
	}

	// Check there is a much distinct mapping than distinct values.
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
