/**
 * \file conv_utf16_align.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#define SIMULATE_PIPELINE
#include <pvkernel/core/picviz_intrin.h>
#include <pvkernel/filter/PVChunkFilter.h>
#include <pvkernel/rush/PVInputFile.h>
#include <pvkernel/rush/PVChunkAlign.h>
#include <pvkernel/rush/PVChunkAlignUTF16Char.h>
#include <pvkernel/rush/PVChunkTransformUTF16.h>
#include <pvkernel/rush/PVRawSource.h>
#include <pvkernel/rush/PVUnicodeSource.h>
#include <pvkernel/rush/PVAggregator.h>
#include <cstdlib>
#include <iostream>
#include "helpers.h"

using std::cout;
using std::cerr;
using std::endl;

using namespace PVRush;
using namespace PVCore;

int main(int argc, char** argv)
{
	if (argc <= 2) {
		cerr << "Usage: " << argv[0] << " file chunk_size" << endl;
		return 1;
	}

	PVCore::PVIntrinsics::init_cpuid();
	PVInput_p ifile(new PVInputFile(argv[1]));
	PVFilter::PVChunkFilter null;
	PVUnicodeSource<> source(ifile, atoi(argv[2]), null);

	PVChunk* pc = source();

	while (pc) {
		dump_chunk_newline(*pc);
		pc->free();
		pc = source();
	}

	return 0;
}
