/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
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

static constexpr const char* csv_file = TEST_FOLDER "/picviz/integer_default_mapping.csv";
static constexpr const char* csv_file_format = TEST_FOLDER "/picviz/integer_default_log.csv.format";

int main()
{
	pvtest::TestEnv env(csv_file, csv_file_format, dupl, pvtest::ProcessUntil::Mapped);

	auto start = std::chrono::system_clock::now();

	Inendi::PVPlotted& plotted = env.compute_plotting();

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;

	std::cout << diff.count();

#ifndef INSPECTOR_BENCH
	// Check mapping is the same as NRaw value.
	PVRush::PVNraw const& nraw = env.root.get_children<Inendi::PVSource>().front()->get_rushnraw();
	const pvcop::db::array& column = nraw.collection().column(0);
	auto& column_array = column.to_core_array<int32_t>();

	std::vector<size_t> order(column.size());
	std::iota(order.begin(), order.end(), 0);
	std::sort(order.begin(), order.end(),
	          [&](size_t a, size_t b) { return column_array[a] > column_array[b]; });

	PV_VALID(plotted.get_column_pointer(0)[order[order.size() - 1]],
	         std::numeric_limits<uint32_t>::max());
	PV_VALID(plotted.get_column_pointer(0)[order[0]], (uint32_t)0);

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
