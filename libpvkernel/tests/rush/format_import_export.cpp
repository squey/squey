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

	std::vector<std::string> input_files(1, argv[1]);
	const char* format = argv[3];
	size_t begin = 0;
	if (argc > 4) {
		errno = 0;
		begin = std::strtoul(argv[4], nullptr, 10);
		if (errno == ERANGE) {
			throw std::runtime_error("Invalid input for begin value");
		}
	}
	if (argc > 5) {
		input_files.emplace_back(argv[5]); // extra input
	}

	pvtest::TestEnv env(input_files, format, DUPL);

	auto start = std::chrono::system_clock::now();

	env.load_data(begin);

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;
	std::cout << diff.count();

#ifndef INSPECTOR_BENCH
	const char* output_file = argv[2];

	if (env._nraw.get_valid_row_count() == 0) {
		PV_VALID(std::string(output_file), std::string("null"));
		return 0;
	}

	std::string output_tmp_file = pvtest::get_tmp_filename();

	// Dump the NRAW to file and check value is the same
	env._nraw.dump_csv(output_tmp_file);

	PV_VALID(PVCore::file_content(output_tmp_file), PVCore::file_content(output_file));

	std::remove(output_tmp_file.c_str());
#endif

	return 0;
}
