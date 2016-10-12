/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#include <inendi/PVPlotted.h>
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
		std::cerr << "Usage: " << argv[0] << " input_file format" << std::endl;
		return 1;
	}

	const char* input_file = argv[1];
	const char* format = argv[2];

	pvtest::TestEnv env(input_file, format, dupl, pvtest::ProcessUntil::Mapped);

	auto start = std::chrono::system_clock::now();

	Inendi::PVPlotted const& plotted = env.compute_plotting();

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;

	std::cout << diff.count();

#ifndef INSPECTOR_BENCH
	// Check mapping is the same as NRaw value.
	PVRush::PVNraw const& nraw = env.root.get_children<Inendi::PVSource>().front()->get_rushnraw();
	const pvcop::db::array& column = nraw.collection().column(0);

	pvcop::db::indexes indexes = pvcop::db::indexes::parallel_sort(column);
	auto& order = indexes.to_core_array();
	std::reverse(order.begin(), order.end()); // plotteds are still inverted...

	PV_VALID(plotted.get_column_pointer(0)[order[order.size() - 1]],
	         std::numeric_limits<uint32_t>::max());
	PV_ASSERT_VALID(plotted.get_column_pointer(0)[order[0]] <= (uint32_t)1);

	// Check we keep value ordering.
	for (size_t i = 1; i < column.size(); i++) {
		PV_ASSERT_VALID(plotted.get_column_pointer(0)[order[i - 1]] <=
		                plotted.get_column_pointer(0)[order[i]]);
	}

#else
	(void)plotted;
#endif

	return 0;
}
