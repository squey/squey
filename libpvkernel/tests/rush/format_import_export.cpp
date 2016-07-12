/**
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2015-2015
 */
#include "common.h"

#include <pvkernel/core/inendi_assert.h>
#include <pvkernel/core/PVUtils.h>

#ifdef INSPECTOR_BENCH
constexpr size_t DUPL = 10000;
#else
constexpr size_t DUPL = 1;
#endif

int main(int argc, char** argv)
{
	if (argc < 4) {
		std::cerr << "Usage: " << argv[0] << " input_file output_file format" << std::endl;
		return 1;
	}

	const char* input_file = argv[1];
	const char* output_file = argv[2];
	const char* format = argv[3];

	static size_t row_count = PVCore::row_count(input_file);

	pvtest::TestEnv env(input_file, format, DUPL);

	auto start = std::chrono::system_clock::now();

	env.load_data(row_count);

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;
	std::cout << diff.count();

#ifndef INSPECTOR_BENCH
	std::string output_tmp_file = pvtest::get_tmp_filename();

	// Dump the NRAW to file and check value is the same
	env._ext.get_nraw().dump_csv(output_tmp_file);

	PV_VALID(PVCore::file_content(output_tmp_file), PVCore::file_content(output_file));

	std::remove(output_tmp_file.c_str());
#endif

	return 0;
}
