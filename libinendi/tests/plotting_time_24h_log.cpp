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
	const pvcop::db::array& column = nraw.collection().column(0);
	auto& array = column.to_core_array<uint64_t>();

	std::vector<uint32_t> order(column.size());
	std::iota(order.begin(), order.end(), 0);

	std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
		const boost::posix_time::ptime ta =
		    *reinterpret_cast<const boost::posix_time::ptime*>(&array[a]);
		const boost::posix_time::ptime tb =
		    *reinterpret_cast<const boost::posix_time::ptime*>(&array[b]);
		tm tm_a = to_tm(ta);
		tm tm_b = to_tm(tb);
		return tm_a.tm_hour > tm_b.tm_hour or
		       (tm_a.tm_hour == tm_b.tm_hour and tm_a.tm_min > tm_b.tm_min) or
		       (tm_a.tm_hour == tm_b.tm_hour and tm_a.tm_min == tm_b.tm_min and
		        tm_a.tm_sec > tm_b.tm_sec);
	});

	uint32_t prev = plotted.get_column_pointer(0)[order[0]];
	const double sec_per_24h = std::log2(3600 * 24);
	const double ratio = std::numeric_limits<uint32_t>::max() / sec_per_24h;
	PV_VALID(prev, 0U);
	for (size_t i = 0; i < column.size(); i++) {
		PV_ASSERT_VALID(prev <= plotted.get_column_pointer(0)[order[i]]);
		prev = plotted.get_column_pointer(0)[order[i]];
	}
	PV_VALID(prev, (uint32_t)(std::numeric_limits<uint32_t>::max() - 1 * ratio + 1));

#else
	(void)plotted;
#endif

	return 0;
}
