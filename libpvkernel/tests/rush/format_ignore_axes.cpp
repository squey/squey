/**
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2015-2015
 */
#include "common.h"

#include <pvkernel/core/inendi_assert.h>

constexpr char FILENAME_IN[] = TEST_FOLDER "/picviz/format_ignore_axes.csv.in";
constexpr char FILENAME_OUT[] = TEST_FOLDER "/picviz/format_ignore_axes.csv.out";
constexpr char FORMAT[] = TEST_FOLDER "/picviz/format_ignore_axes.csv.format";

#ifdef INSPECTOR_BENCH
constexpr size_t DUPL = 10000;
#else
constexpr size_t DUPL = 1;
#endif

constexpr static size_t nb_lines = 14 * DUPL;

int main()
{
	pvtest::TestEnv env(FILENAME_IN, FORMAT, nb_lines);

	auto start = std::chrono::system_clock::now();

	env.load_data(nb_lines);

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;
	std::cout << diff.count();

#ifndef INSPECTOR_BENCH
	std::string out_path = pvtest::get_tmp_filename();

	// Dump the NRAW to file and check value is the same
	env._ext.get_nraw().dump_csv(out_path);

	std::ifstream ifs_res(out_path);
	std::string content_res{std::istreambuf_iterator<char>(ifs_res),
	                        std::istreambuf_iterator<char>()};

	std::ifstream ifs_ref(FILENAME_OUT);
	std::string content_ref{std::istreambuf_iterator<char>(ifs_ref),
	                        std::istreambuf_iterator<char>()};

	PV_VALID(content_ref, content_res);

	std::remove(out_path.c_str());
#endif

	return 0;
}
