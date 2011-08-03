#define SIMULATE_PIPELINE
#include <pvkernel/rush/PVInputFile.h>
#include <pvkernel/rush/PVChunkAlign.h>
#include <pvkernel/rush/PVChunkTransform.h>
#include <pvkernel/rush/PVRawSource.h>
#include <cstdlib>
#include <iostream>
#include "helpers.h"

using std::cout;
using std::cerr;
using std::endl;

using namespace PVRush;

int main(int argc, char** argv)
{
	if (argc < 4) {
		cerr << "Usage: " << argv[0] << " file chunk_size nchunks" << endl;
		return 1;
	}

	PVInputFile ifile(argv[1]);
	PVChunkAlign calign;
	PVChunkTransform transform;
	PVRawSource<> source(ifile, calign, atoi(argv[2]), transform);
	const int nchunks = atoi(argv[3]);

	PVChunk* chunks_p[nchunks];

	// Create NCHUNKS chunks using std::allocator
	for (int i = 0; i < nchunks; i++) {
		chunks_p[i] = source();
	}

	// Read them
	for (int i = 0; i < nchunks; i++)
	{
		PVChunk* p = chunks_p[i];
		if (p == NULL)
			continue;
		dump_chunk(*p);
		
		// Deallocate
		p->free();
	}

	return 0;
}
