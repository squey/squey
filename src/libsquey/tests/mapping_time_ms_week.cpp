//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <squey/PVMapped.h>
#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVUtils.h>

#include "common.h"

#include <chrono>
#include <iostream>

#include <unicode/calendar.h>

using namespace icu_75;

#ifdef SQUEY_BENCH
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

	Squey::PVMapped& mapped = env.compute_mapping();

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;

	std::cout << diff.count();

#ifndef SQUEY_BENCH
	// Compute distinct values.
	PVRush::PVNraw const& nraw = env.root.get_children<Squey::PVSource>().front()->get_rushnraw();
	const pvcop::db::array& column = nraw.column(PVCol(0));
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

	uint32_t prev = mapped.get_column(PVCol(0)).to_core_array<uint32_t>()[order[0]];
	for (size_t i = 0; i < column.size(); i++) {
		PV_ASSERT_VALID(prev <= mapped.get_column(PVCol(0)).to_core_array<uint32_t>()[order[i]]);
		prev = mapped.get_column(PVCol(0)).to_core_array<uint32_t>()[order[i]];
	}
#else
	(void)mapped;
#endif

	return 0;
}
