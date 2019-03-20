/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVPlotted.h>
#include <pvkernel/core/inendi_assert.h>
#include <pvkernel/rush/PVUtils.h>

#include "common.h"

#include <boost/date_time/posix_time/posix_time.hpp>

#include <chrono>
#include <iostream>

#ifdef INSPECTOR_BENCH
// 10 000 000 lines.
static constexpr int dupl = 200;
#else
static constexpr int dupl = 1;
#endif

static constexpr const char* csv_file = TEST_FOLDER "/picviz/time_mapping_us.csv";
static constexpr const char* csv_file_format = TEST_FOLDER "/picviz/datetime_us_24h_log.csv.format";

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
	auto& array = column.to_core_array<boost::posix_time::ptime>();

	std::vector<uint32_t> order(column.size());
	std::iota(order.begin(), order.end(), 0);

	std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
		const boost::posix_time::ptime pt1 = array[a];
		const boost::posix_time::ptime pt2 = array[b];
		const auto& t1 = pt1.time_of_day();
		const auto& t2 = pt2.time_of_day();
		return (
		    t1.hours() > t2.hours() or (t1.hours() == t2.hours() and t1.minutes() > t2.minutes()) or
		    (t1.hours() == t2.hours() and t1.minutes() == t2.minutes() and
		     t1.seconds() > t2.seconds()) or
		    (t1.hours() == t2.hours() and t1.minutes() == t2.minutes() and
		     t1.seconds() == t2.seconds() and t1.fractional_seconds() > t2.fractional_seconds()));
	});

	uint32_t prev = plotted.get_column_pointer(PVCol(0))[order[0]];
	for (size_t i = 0; i < column.size(); i++) {
		PV_ASSERT_VALID(prev <= plotted.get_column_pointer(PVCol(0))[order[i]]);
		prev = plotted.get_column_pointer(PVCol(0))[order[i]];
	}
	PV_ASSERT_VALID(plotted.get_column_pointer(PVCol(0))[order[0]] <
	                plotted.get_column_pointer(PVCol(0))[order[column.size() - 1]]);

#else
	(void)plotted;
#endif

	return 0;
}
