/**
 * \file base.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#define SIMULATE_PIPELINE
#include <pvkernel/rush/PVInputFile.h>
#include <pvkernel/rush/PVChunkAlign.h>
#include <pvkernel/rush/PVRawSource.h>
#include <pvkernel/filter/PVChunkFilterByElt.h>
#include <pvkernel/filter/PVElementFilter.h>
#include <pvkernel/filter/PVElementFilterByFields.h>
#include <pvkernel/filter/PVFieldsFilter.h>
#include <cstdlib>
#include <iostream>
#include "helpers.h"

using std::cout;
using std::cerr;
using std::endl;

using namespace PVRush;
using namespace PVFilter;

int main(int argc, char** argv)
{
	if (argc < 4) {
		cerr << "Usage: " << argv[0] << " file chunk_size nchunks" << endl;
		return 1;
	}

	PVInputFile ifile(argv[1]);
	PVChunkAlign calign;
	PVRawSource<> source(ifile, calign, atoi(argv[2]));
	const int nchunks = atoi(argv[3]);
	
	PVChunk* chunks_p[nchunks];

	// Create NCHUNKS chunks using std::allocator
	for (int i = 0; i < nchunks; i++) {
		chunks_p[i] = source();
	}

	// Apply null element filter, followed by null fields filters
	// Using composition
	PVElementFilter elt_flt_null;
	
	PVFieldsFilter<> f_flt_null;
	PVElementFilterByFields elt_flt_fields(f_flt_null.f());

	PVElementFilter_f elt_flt_comp = boost::bind(elt_flt_null.f(), boost::bind(elt_flt_fields.f(), _1));
	//PVChunkFilterByElt chk_flt(elt_flt_comp);
	PVChunkFilterByElt chk_flt(elt_flt_null);
	for (int i = 0; i < nchunks; i++)
	{
		PVChunk* p = chunks_p[i];
		if (p == NULL)
			continue;
		dump_chunk(*p);
		// Filter
		chk_flt(p);
		
		// Deallocate
		p->free();
	}

	return 0;
}
