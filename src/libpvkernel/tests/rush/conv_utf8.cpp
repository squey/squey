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

#include <pvkernel/core/squey_intrin.h>
#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVInputFile.h>
#include <pvkernel/rush/PVUtils.h>
#include <pvkernel/rush/PVUnicodeSource.h>
#include <iostream>

#include "common.h"
#include "helpers.h"

using namespace PVRush;
using namespace PVCore;

static constexpr const char* input_file = TEST_FOLDER "/pvkernel/rush/charset/utf8";
#ifndef SQUEY_BENCH
static constexpr const char* ref_file = TEST_FOLDER "/pvkernel/rush/charset/utf8.out";
#endif

int main(int argc, char** argv)
{
	const char* input = (argc < 4) ? input_file : argv[1];
#ifndef SQUEY_BENCH
	const char* ref = (argc < 4) ? ref_file : argv[2];
#endif
	const size_t chunk_size = (argc < 4) ? 20000 : atoi(argv[3]);

	pvtest::init_ctxt();

	PVInput_p ifile(new PVInputFile(input));
	PVUnicodeSource<> source(ifile, chunk_size);

	std::string output_file = pvtest::get_tmp_filename();
	// Extract source and split fields.
	{
		std::ofstream ofs{std::filesystem::path(output_file)};

		std::chrono::duration<double> dur(0.);
		auto start = std::chrono::steady_clock::now();
		while (PVCore::PVTextChunk* pc = source()) {
			auto end = std::chrono::steady_clock::now();
			dur += end - start;
			dump_chunk_csv(*pc, ofs);
			pc->free();
			start = std::chrono::steady_clock::now();
		}
		std::cout << dur.count();
	}

#ifndef SQUEY_BENCH
	// Check output is the same as the reference
	std::cout << std::endl << output_file << " - " << ref << std::endl;
	PV_ASSERT_VALID(PVRush::PVUtils::files_have_same_content(output_file, ref));
#endif

	std::remove(output_file.c_str());

	return 0;
}
