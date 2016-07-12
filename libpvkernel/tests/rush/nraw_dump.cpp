/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2015
 */

#include <fstream>

#include <pvkernel/core/inendi_assert.h>

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

	env._ext.get_nraw().dump_csv(out_path);

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;
	std::cout << diff.count();

#ifndef INSPECTOR_BENCH
	std::ifstream ifs_res(out_path);
	std::string content_res{std::istreambuf_iterator<char>(ifs_res),
	                        std::istreambuf_iterator<char>()};

	std::ifstream ifs_ref(filename);
	std::string content_ref{std::istreambuf_iterator<char>(ifs_ref),
	                        std::istreambuf_iterator<char>()};

	PV_VALID(content_ref, content_res);
#endif

	std::remove(out_path.c_str());

	return 0;
}
