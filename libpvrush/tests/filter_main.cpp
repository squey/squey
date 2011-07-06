#define SIMULATE_PIPELINE
#include <pvfilter/PVChunkFilterByElt.h>
#include <pvrush/PVInputFile.h>
#include <pvrush/PVChunkAlign.h>
#include <pvrush/PVRawSource.h>
#include <cstdlib>
#include <iostream>
#include "helpers.h"
#include "filter_main.h"

#ifdef PV_USE_RINGBUFFER
#include <pvrush/PVVrbAllocator.h>
#endif

using std::cout;
using std::cerr;
using std::endl;

using namespace PVRush;
using namespace PVFilter;
using namespace PVCore;

#ifdef PV_USE_RINGBUFFER
typedef PVUnicodeSource<PVVrbAllocator> PVUnicodeSourceRB;
#endif

void chunk_filter_test(PVChunkFilter_f chk_flt, int argc, char** argv)
{
	if (argc <= 2) {
		cerr << "Usage: " << argv[0] << " file chunk_size" << endl;
		std::exit(1);
	}

	PVInputFile ifile(argv[1]);
	PVChunkFilter null;
	
	const int chunk_size = atoi(argv[2]);
#ifdef PV_USE_RINGBUFFER
	vrb_p v = vrb_new(nchunks*(chunk_size*sizeof(char)+100), NULL);
	PVawSourceRB::alloc_chunk alloc(v);	
	PVUnicodeSourceRB source(ifile, chunk_size, null, alloc);
#else
	PVUnicodeSource<> source(ifile, chunk_size, null);
#endif
	
	PVChunk* pc = source();
	while (pc)
	{
		// Filter
		chk_flt(pc);
		dump_chunk_csv(*pc);
		
		// Deallocate
		pc->free();
		pc = source();
	}

#ifdef PV_USE_RINGBUFFER
	vrb_destroy(v);
#endif
}
