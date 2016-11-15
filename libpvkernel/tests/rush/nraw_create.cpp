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
 * Load data in the NRaw.
 */
int main()
{
	pvtest::TestEnv env(filename, fileformat, nb_dup);

	auto start = std::chrono::system_clock::now();

	env.load_data();

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;
	std::cout << diff.count();

#ifndef INSPECTOR_BENCH
	std::string out_path = pvtest::get_tmp_filename();
	// Dump the NRAW to file and check value is the same
	PVRush::PVNraw nraw = std::move(env._nraw);
	nraw.dump_csv(out_path);

	PV_ASSERT_VALID(PVRush::PVUtils::files_have_same_content(filename, out_path));

	std::remove(out_path.c_str());
#endif

	return 0;
}
