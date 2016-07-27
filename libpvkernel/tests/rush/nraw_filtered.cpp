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
const std::string fileformat = TEST_FOLDER "/picviz/filtered.csv.format";

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

	PV_VALID(env.get_nraw_size(), 0UL);

	return 0;
}
