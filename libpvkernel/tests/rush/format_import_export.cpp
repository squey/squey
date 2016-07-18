/**
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2015-2015
 */
#include "common.h"

#include <pvkernel/core/inendi_assert.h>
#include <pvkernel/core/PVUtils.h>

#include <sys/stat.h>

#ifdef INSPECTOR_BENCH
constexpr size_t DUPL = 10000;
#else
constexpr size_t DUPL = 1;
#endif

int main(int argc, char** argv)
{
	if (argc < 4) {
		std::cerr << "Usage: " << argv[0] << " input_file output_file format [begin] [extra_input]"
		          << std::endl;
		return 1;
	}

	const char* input_file = argv[1];
	const char* output_file = argv[2];
	const char* format = argv[3];
	size_t begin = 0;
	if (argc > 4) {
		errno = 0;
		begin = std::strtoul(argv[4], nullptr, 10);
		if (errno == ERANGE) {
			throw std::runtime_error("Invalid input for begin value");
		}
	}
	std::string extra_input;
	if (argc > 5) {
		extra_input = argv[5];
	}

	pvtest::TestEnv env(input_file, format, DUPL, extra_input);

	auto start = std::chrono::system_clock::now();

	env.load_data(begin);

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
