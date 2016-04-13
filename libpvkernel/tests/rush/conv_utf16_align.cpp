/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/inendi_intrin.h>
#include <pvkernel/rush/PVInputFile.h>
#include <pvkernel/rush/PVUnicodeSource.h>
#include <pvkernel/rush/PVAggregator.h>
#include <iostream>
#include "helpers.h"

using namespace PVRush;
using namespace PVCore;

int main(int argc, char** argv)
{
	if (argc <= 2) {
		std::cerr << "Usage: " << argv[0] << " file chunk_size" << std::endl;
		return 1;
	}

	PVCore::PVIntrinsics::init_cpuid();
	PVInput_p ifile(new PVInputFile(argv[1]));
	PVUnicodeSource<> source(ifile, atoi(argv[2]));

	while (PVChunk* pc = source()) {
		dump_chunk_raw(*pc);
		pc->free();
	}

	return 0;
}
