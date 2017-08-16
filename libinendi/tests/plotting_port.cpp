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
using plotting_t = Inendi::PVPlottingFilter::value_type;
static constexpr const char* csv_file = TEST_FOLDER "/picviz/plotting_port.csv";
static constexpr const char* csv_file_format = TEST_FOLDER "/picviz/plotting_port.csv.format";
static constexpr const plotting_t threshold1 = (0.3 * std::numeric_limits<plotting_t>::max()) - 1;
static constexpr const plotting_t threshold2 = (0.6 * std::numeric_limits<plotting_t>::max()) - 1;
static constexpr const plotting_t max_threshold = std::numeric_limits<plotting_t>::max();

int main()
{
	pvtest::TestEnv env(csv_file, csv_file_format, dupl, pvtest::ProcessUntil::Mapped);

	auto start = std::chrono::system_clock::now();

	Inendi::PVPlotted const& plotted = env.compute_plotting();

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;

	std::cout << diff.count();

#ifndef INSPECTOR_BENCH
	// Compute distinct values.
	PVRush::PVNraw const& nraw = env.root.get_children<Inendi::PVSource>().front()->get_rushnraw();
	const pvcop::db::array& column = nraw.column(PVCol(0));
	auto& array = column.to_core_array<int32_t>();

	for (size_t i = 1; i < array.size(); i++) {
		PV_ASSERT_VALID(plotted.get_column_pointer(PVCol(0))[i] <
		                plotted.get_column_pointer(PVCol(0))[i - 1]);
		if (i < 1024) {
			PV_ASSERT_VALID(~plotting_t(plotted.get_column_pointer(PVCol(0))[i]) <= threshold1);
		} else if (i >= 1024 && i <= 49151) {
			PV_ASSERT_VALID(~plotting_t(plotted.get_column_pointer(PVCol(0))[i]) > threshold1 &&
			                ~plotting_t(plotted.get_column_pointer(PVCol(0))[i]) <= threshold2);
		} else {

			PV_ASSERT_VALID(~plotting_t(plotted.get_column_pointer(PVCol(0))[i]) > threshold2 &&
			                ~plotting_t(plotted.get_column_pointer(PVCol(0))[i]) <= max_threshold);
		}
	}

#else
	(void)plotted;
	(void)threshold1;
	(void)threshold2;
	(void)max_threshold;
#endif

	return 0;
}
