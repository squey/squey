/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2015
 */

#include <fstream>

#include <pvkernel/core/inendi_assert.h>

#include <pvkernel/rush/PVUtils.h>

#include "common.h"

#ifdef INSPECTOR_BENCH
constexpr static size_t nb_dup = 20;
#else
constexpr static size_t nb_dup = 1;
#endif

const std::string filename = TEST_FOLDER "/picviz/heat_line.csv";
const std::string fileformat = TEST_FOLDER "/picviz/heat_line.csv.format";

/**
 * Load an NRaw from a file and a format, dump it and check result is the same as csv input.
 */
int main()
{
	pvtest::TestEnv env(filename, fileformat, nb_dup);

	env.load_data();

	// Dump the NRAW to file and check value is the same
	auto start = std::chrono::system_clock::now();

	std::string out_path = pvtest::get_tmp_filename();

	env._nraw.dump_csv(out_path);

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;
	std::cout << diff.count();

#ifndef INSPECTOR_BENCH
	PV_ASSERT_VALID(PVRush::PVUtils::files_have_same_content(filename, out_path));
#endif

	std::remove(out_path.c_str());

	return 0;
}
