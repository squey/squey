#define SIMULATE_PIPELINE
#include <pvrush/PVInputFile.h>
#include <pvrush/PVChunkAlign.h>
#include <pvrush/PVChunkTransformUTF16.h>
#include <pvrush/PVRawSource.h>
#include <pvfilter/PVChunkFilter.h>
#include <cstdlib>
#include <iostream>
#include "helpers.h"
#include <QCoreApplication>

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

	PVInput_p ifile(new PVInputFile(argv[1]));
	PVChunkAlign calign;
	PVChunkTransformUTF16 transform;
	PVFilter::PVChunkFilter null;
	PVRawSource<> source(ifile, calign, atoi(argv[2]), transform, null);

	PVChunk* pc = source();

	while (pc) {
		dump_chunk_raw(*pc);
		pc->free();
		pc = source();
	}

	return 0;
}
