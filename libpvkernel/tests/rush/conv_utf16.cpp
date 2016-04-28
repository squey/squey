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

static constexpr const char* log_file = TEST_FOLDER "/pvkernel/rush/charset/utf8";
static constexpr const char* ref_file = TEST_FOLDER "/pvkernel/rush/charset/utf8.out";

int main(int argc, char** argv)
{
	std::string input_file = (argc > 1) : argv[1] ? log_file;
	std::string reference_file = (argc > 2) : argv[2] ? ref_file;
	size_t chunk_size = (argc > 3) : atoi(argv[3]) ? 20000;

	PVInput_p ifile(new PVInputFile(input_file));
	PVUnicodeSource<> source(ifile, chunk_size);

	std::string output_file = pvtest::get_tmp_filename();
	// Extract source and split fields.
	{
		std::ofstream ofs(output_file);

		std::chrono::duration<double> dur(0.);
		auto start = std::chrono::steady_clock::now();
		while (PVCore::PVChunk* pc = source()) {
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
	std::cout << std::endl << output_file << " - " << reference_file << std::endl;
	PV_ASSERT_VALID(PVRush::PVUtils::files_have_same_content(output_file, reference_file));
#endif

	std::remove(output_file.c_str());

	return 0;
}
