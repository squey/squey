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

#include <unicode/calendar.h>

#ifdef INSPECTOR_BENCH
// 10 000 000 lines.
static constexpr int dupl = 200;
#else
static constexpr int dupl = 1;
#endif

static constexpr const char* csv_file = TEST_FOLDER "/picviz/time_mapping.csv";
static constexpr const char* csv_file_format =
    TEST_FOLDER "/picviz/datetime_ms_week_mapping.csv.format";

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
	PVRush::PVNraw const& nraw = env.root->get_children<Inendi::PVSource>().front()->get_rushnraw();
	const pvcop::db::array& column = nraw.collection().column(0);
	auto& array = column.to_core_array<uint64_t>();

	std::vector<uint32_t> order(column.size());
	std::iota(order.begin(), order.end(), 0);

	UErrorCode err = U_ZERO_ERROR;
	std::unique_ptr<Calendar> cal(Calendar::createInstance(err));
	if (not U_SUCCESS(err)) {
		return -1;
	}
	std::unique_ptr<Calendar> cal_2(Calendar::createInstance(err));
	if (not U_SUCCESS(err)) {
		return -1;
	}

	std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
		cal->setTime(static_cast<UDate>(array[a]), err);
		cal_2->setTime(static_cast<UDate>(array[b]), err);
		return cal->get(UCAL_DAY_OF_WEEK, err) < cal_2->get(UCAL_DAY_OF_WEEK, err) or
		       (cal->get(UCAL_DAY_OF_WEEK, err) == cal_2->get(UCAL_DAY_OF_WEEK, err) and
		        cal->get(UCAL_HOUR_OF_DAY, err) < cal_2->get(UCAL_HOUR_OF_DAY, err)) or
		       (cal->get(UCAL_DAY_OF_WEEK, err) == cal_2->get(UCAL_DAY_OF_WEEK, err) and
		        cal->get(UCAL_HOUR_OF_DAY, err) == cal_2->get(UCAL_HOUR_OF_DAY, err) and
		        cal->get(UCAL_MINUTE, err) < cal_2->get(UCAL_MINUTE, err)) or
		       (cal->get(UCAL_DAY_OF_WEEK, err) == cal_2->get(UCAL_DAY_OF_WEEK, err) and
		        cal->get(UCAL_HOUR_OF_DAY, err) == cal_2->get(UCAL_HOUR_OF_DAY, err) and
		        cal->get(UCAL_MINUTE, err) == cal_2->get(UCAL_MINUTE, err) and
		        cal->get(UCAL_SECOND, err) < cal_2->get(UCAL_SECOND, err));
	});

	uint32_t prev = mapped->get_value(order[0], 0).storage_as_uint();
	for (size_t i = 0; i < column.size(); i++) {
		PV_ASSERT_VALID(prev <= mapped->get_value(order[i], 0).storage_as_uint());
		prev = mapped->get_value(order[i], 0).storage_as_uint();
	}
#endif

	return 0;
}
