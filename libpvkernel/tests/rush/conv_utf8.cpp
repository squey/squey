/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#include <pvkernel/core/inendi_intrin.h>
#include <pvkernel/core/inendi_assert.h>
#include <pvkernel/rush/PVInputFile.h>
#include <pvkernel/rush/PVUtils.h>
#include <pvkernel/rush/PVUnicodeSource.h>
#include <iostream>

#include "common.h"
#include "helpers.h"

using namespace PVRush;
using namespace PVCore;

static constexpr const char* input_file = TEST_FOLDER "/pvkernel/rush/charset/utf8";
#ifndef INSPECTOR_BENCH
static constexpr const char* ref_file = TEST_FOLDER "/pvkernel/rush/charset/utf8.out";
#endif

int main(int argc, char** argv)
{
	const char* input = (argc < 4) ? input_file : argv[1];
#ifndef INSPECTOR_BENCH
	const char* ref = (argc < 4) ? ref_file : argv[2];
#endif
	const size_t chunk_size = (argc < 4) ? 20000 : atoi(argv[3]);

	pvtest::init_ctxt();

	PVInput_p ifile(new PVInputFile(input));
	PVUnicodeSource<> source(ifile, chunk_size);

	std::string output_file = pvtest::get_tmp_filename();
	// Extract source and split fields.
	{
		std::ofstream ofs(output_file);

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

#ifndef INSPECTOR_BENCH
	// Check output is the same as the reference
	std::cout << std::endl << output_file << " - " << ref << std::endl;
	PV_ASSERT_VALID(PVRush::PVUtils::files_have_same_content(output_file, ref));
#endif

	std::remove(output_file.c_str());

	return 0;
}
