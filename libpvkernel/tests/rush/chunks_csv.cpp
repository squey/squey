#define SIMULATE_PIPELINE
#include <pvkernel/filter/PVChunkFilter.h>
#include <pvkernel/filter/PVChunkFilterSource.h>
#include <pvkernel/rush/PVInputFile.h>
#include <pvkernel/rush/PVChunkAlign.h>
#include <pvkernel/rush/PVChunkAlignUTF16Char.h>
#include <pvkernel/rush/PVChunkTransformUTF16.h>
#include <pvkernel/rush/PVRawSource.h>
#include <pvkernel/rush/PVAggregator.h>
#include <pvkernel/filter/PVFieldSplitterCSV.h>
#include <pvkernel/filter/PVElementFilterByFields.h>
#include <pvkernel/filter/PVChunkFilterByElt.h>
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
	if (argc < 4) {
		cerr << "Usage: " << argv[0] << " file chunk_size nchunks" << endl;
		return 1;
	}

	PVInputFile ifile(argv[1]);
	PVFilter::PVChunkFilter filter_null;
	PVChunkTransformUTF16 transform;
	PVFilter::PVFieldSplitterCSV csv;
	PVFilter::PVElementFilterByFields field_f(csv.f());
	PVFilter::PVChunkFilterByElt chk_flt(field_f.f());
	PVFilter::PVChunkFilterSource src_filter;
	PVChunkAlignUTF16Char calign(QChar('\n'));

	PVRawSource<> source(ifile, calign, atoi(argv[2]), transform, chk_flt);
	PVAggregator agg = PVAggregator::from_unique_source(&source);
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
		p = src_filter(p);
		dump_chunk(*p);
		cout << " -- " << endl << endl;
		
		// Deallocate
		p->free();
	}

	return 0;
}
