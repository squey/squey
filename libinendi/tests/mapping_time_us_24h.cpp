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
static constexpr const char* csv_file_format =
    TEST_FOLDER "/picviz/datetime_us_24h_mapping.csv.format";

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
	const pvcop::db::array& column = nraw.column(0);
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
		return tm_a.tm_hour < tm_b.tm_hour or
		       (tm_a.tm_hour == tm_b.tm_hour and tm_a.tm_min < tm_b.tm_min) or
		       (tm_a.tm_hour == tm_b.tm_hour and tm_a.tm_min == tm_b.tm_min and
		        tm_a.tm_sec < tm_b.tm_sec);
	});

	uint32_t prev = mapped.get_column(0).to_core_array<uint32_t>()[order[0]];
	for (size_t i = 0; i < column.size(); i++) {
		PV_ASSERT_VALID(prev <= mapped.get_column(0).to_core_array<uint32_t>()[order[i]]);
		prev = mapped.get_column(0).to_core_array<uint32_t>()[order[i]];
	}
#else
	(void)mapped;
#endif

	return 0;
}
