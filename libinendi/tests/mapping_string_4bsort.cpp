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

static constexpr const char* csv_file = TEST_FOLDER "/picviz/string_mapping.csv";
static constexpr const char* csv_file_format = TEST_FOLDER "/picviz/string_mapping.csv.format";
#ifndef INSPECTOR_BENCH
static constexpr const char* ref_file = TEST_FOLDER "/picviz/string_mapping_4bsort.ref";
#endif

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

	std::string res_file = pvtest::get_tmp_filename();
	std::ofstream ofs(res_file);

	for (size_t i = 0; i < column.size(); i++) {
		ofs << mapped->get_value(i, 0).storage_as_uint() << std::endl;
	}

	std::cout << res_file << "/" << ref_file << std::endl;
	PV_ASSERT_VALID(PVRush::PVUtils::files_have_same_content(res_file, ref_file));

	std::remove(res_file.c_str());
#endif

	return 0;
}
