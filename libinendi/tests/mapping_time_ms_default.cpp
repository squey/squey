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

static constexpr const char* csv_file = TEST_FOLDER "/picviz/time_mapping.csv";
static constexpr const char* csv_file_format = TEST_FOLDER "/picviz/datetime_ms_default_mapping.csv.format";

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
    PVRush::PVNraw const& nraw = env.src->get_rushnraw();
    const pvcop::db::array& column = nraw.collection().column(0);
    auto& array = column.to_core_array<uint64_t>();

    std::vector<uint32_t> order(column.size());
    std::iota(order.begin(), order.end(), 0);

    std::sort(order.begin(), order.end(), [&array](uint32_t a, uint32_t b) { return (array[a] / 1000) < (array[b] / 1000); });

    uint32_t prev = mapped->get_value(order[0], 0).storage_as_uint();
    for(size_t i=0; i<column.size(); i++) {
	PV_ASSERT_VALID(prev <= mapped->get_value(order[i], 0).storage_as_uint());
	prev = mapped->get_value(order[i], 0).storage_as_uint();
    }
#endif

    return 0;
}
