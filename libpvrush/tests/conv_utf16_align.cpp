#define SIMULATE_PIPELINE
#include <pvfilter/PVChunkFilter.h>
#include <pvrush/PVInputFile.h>
#include <pvrush/PVChunkAlign.h>
#include <pvrush/PVChunkAlignUTF16Char.h>
#include <pvrush/PVChunkTransformUTF16.h>
#include <pvrush/PVRawSource.h>
#include <pvrush/PVUnicodeSource.h>
#include <pvrush/PVAggregator.h>
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

	PVInputFile ifile(argv[1]);
	PVFilter::PVChunkFilter null;
	PVUnicodeSource<> source(ifile, atoi(argv[2]), null);

	PVChunk* pc = source();

	while (pc) {
		dump_chunk_csv(*pc);
		pc->free();
		pc = source();
	}

	return 0;
}
