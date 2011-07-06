#include <pvrush/PVInputFile.h>
#include <pvrush/PVAggregator.h>
#include <pvrush/PVChunkAlignUTF16Char.h>
#include <pvrush/PVChunkTransformUTF16.h>
#include <pvrush/PVRawSource.h>
#include <pvrush/PVUnicodeSource.h>
#include <pvrush/PVOutputFile.h>
#include <pvrush/PVNrawOutput.h>
#include <pvrush/PVVrbAllocator.h>
#include <pvfilter/PVChunkFilterCountElts.h>
#include <pvfilter/PVChunkFilterSource.h>
#include <pvcore/debug.h>
#include "helpers.h"
#include "filter_main.h"

// STL
#include <cstdlib>
#include <string>
#include <iostream>

// TBB
#include <tbb/pipeline.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/tick_count.h>

// VRB (ring buffer)
#ifdef PV_USE_RINGBUFFER
#include <vrb.h>
#endif

using namespace PVFilter;
using namespace PVRush;
using namespace PVCore;

#ifdef PV_USE_RINGBUFFER
typedef PVRawSource<PVVrbAllocator> PVRawSourceRB;
#endif

void chunk_filter_test(PVChunkFilterByElt &chk_flt, int argc, char** argv)
{
	if (argc < 5) {
		std::cerr << "Usage: " << argv[0] << " file chunk_size nchunks nelts" << std::endl;
		std::exit(1);
	}

	PVInputFile ifile(argv[1]);
	PVChunkFilter null;
	bool done = false;
	PVChunkFilterCountElts count(&done);
	PVChunkFilterSource src_flt;
	count.done_when(atoi(argv[4]));
	const int nchunks = atoi(argv[3]);
	const int chunk_size = atoi(argv[2]);
#ifdef PV_USE_RINGBUFFER
	vrb_p v = vrb_new(nchunks*(chunk_size*sizeof(char)+100), NULL);
	PVRawSourceRB::alloc_chunk alloc(v);	
	PVRawSourceBase_p source(new PVRawSourceRB(ifile, calign, chunk_size, transform, null, alloc));
#else
	PVRawSourceBase_p source(new PVUnicodeSource<>(ifile, chunk_size, null));
#endif
	PVAggregator agg = PVAggregator::from_unique_source(source);
	agg.set_stop_condition(&done);


	PVRush::PVNraw *nraw = new PVRush::PVNraw;
	PVNrawOutput onraw(*nraw);

	std::string out_path = argv[1];
	out_path += ".out";
	PVOutputFile ofile(out_path.c_str());

	// Create TBB filters
	tbb::filter_t<void,PVChunk*> input_filter(tbb::filter::serial_in_order, agg);
	tbb::filter_t<PVChunk*, PVChunk*> source_filter(tbb::filter::parallel, src_flt.f());
	tbb::filter_t<PVChunk*, PVChunk*> transform_filter(tbb::filter::parallel, chk_flt.f());
	tbb::filter_t<PVChunk*, PVChunk*> count_filter(tbb::filter::serial_in_order, count.f());
	tbb::filter_t<PVChunk*,void> output_filter(tbb::filter::serial_in_order, onraw.f());
//	tbb::filter_t<PVChunk*,void> output_filter(tbb::filter::serial_in_order, ofile.f());

	// And launch everything

	// Number of threads is automatic
	tbb::task_scheduler_init init_parallel(1);

	tbb::tick_count t0 = tbb::tick_count::now();
	tbb::parallel_pipeline(nchunks, input_filter & source_filter & transform_filter & count_filter & output_filter);
	tbb::tick_count t1 = tbb::tick_count::now();
	
	std::cout << "parallel run time = " << (t1-t0).seconds() << std::endl;
	std::cout << "Is job completed: " << done << std::endl;

	std::cout << "NRaw output" << std::endl;
	for (int i = 0; i < nraw->table.size(); i++) {
		std::cout << "Line " << i << " ";
		//::debug_qstringlist(nraw->table[i]);
	}

	delete nraw;
}
