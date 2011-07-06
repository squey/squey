#include <pvrush/PVInputFile.h>
#include <pvrush/PVChunkAlignUTF16Char.h>
#include <pvrush/PVChunkTransformUTF16.h>
#include <pvrush/PVRawSource.h>
#include <pvrush/PVOutputFile.h>
#include <pvrush/PVVrbAllocator.h>
#include <pvrush/PVExtractor.h>
#include <pvrush/PVControllerJob.h>
#include <pvfilter/PVChunkFilter.h>
#include <pvfilter/PVElementFilterRandInvalid.h>
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
#include <tbb/compat/thread>

// VRB (ring buffer)
#ifdef PV_USE_RINGBUFFER
#include <vrb.h>
#endif

using namespace PVFilter;
using namespace PVRush;

#ifdef PV_USE_RINGBUFFER
typedef PVRawSource<PVVrbAllocator> PVRawSourceRB;
#endif

void dump_nraw(PVNraw const& nraw)
{
	for (int i = 0; i < nraw.table.size(); i++) {
		std::cout << "Line " << i << ": ";
		//::debug_qstringlist(nraw.table[i]);
	}
}

void chunk_filter_test(PVChunkFilterByElt &chk_flt, int argc, char** argv)
{
	if (argc <= 2) {
		std::cerr << "Usage: " << argv[0] << " chunk_size files" << std::endl;
		std::exit(1);
	}

	PVElementFilterRandInvalid elt_f_invalid;
	PVChunkFilterByElt chk_invalid(elt_f_invalid.f());

	PVExtractor ext;
	ext.start_controller();
	ext.set_chunk_filter(boost::bind(chk_flt.f(), boost::bind(chk_invalid.f(), _1)));
	//ext.set_chunk_filter(chk_flt.f());

	PVChunkAlignUTF16Char calign(QChar('\n'));
	PVChunkTransformUTF16 transform;
	PVChunkFilter null;
	const int chunk_size = atoi(argv[1]);
	for (int i = 2; i < argc; i++) {
		PVInputFile *ifile = new PVInputFile(argv[i]);
#ifdef PV_USE_RINGBUFFER
		vrb_p v = vrb_new(nchunks*(chunk_size*sizeof(char)+100), NULL);
		PVRawSourceRB::alloc_chunk alloc(v);	
		PVRawSourceBase_p source(new PVRawSourceRB(ifile, calign, chunk_size, transform, null, alloc));
#else
		PVRawSourceBase_p source(new PVRawSource<>(*ifile, calign, chunk_size, transform, null));
#endif
		ext.add_source(source);
	}

	// Use the extractor in different ways !
	
	// 100 lines from the beggining
	std::cout << "Ask 100 lines from line 0:" << std::endl;
	PVControllerJob_p job = ext.process_from_agg_nlines(0, 100);
	job->wait_end();
	std::cout << "Done. Nraw dump :" << std::endl;
	dump_nraw(ext.get_nraw());

	// 5 lines from the 10th one
	std::cout << "Ask 50 lines from line 10 :" << std::endl;
	job = ext.process_from_agg_nlines(10, 50);
	job->wait_end();
	std::cout << "Done. Nraw dump :" << std::endl;
	dump_nraw(ext.get_nraw());

	// 50 lines from the 100th one
	std::cout << "Ask 100 lines from line 50 :" << std::endl;
	job = ext.process_from_agg_nlines(50, 100);
	job->wait_end();
	std::cout << "Done. Nraw dump :" << std::endl;
	dump_nraw(ext.get_nraw());

	// The nraw has got 50 lines. Shrink it to 20
	std::cout << "Ask the nraw from line 0 to 24 :" << std::endl;
	job = ext.process_from_pvrow(0, 24, 0, false);
	job->wait_end();
	std::cout << "Done. Nraw dump :" << std::endl;
	dump_nraw(ext.get_nraw());

	// Now ask for nraw lines from 0 to 40
	std::cout << "Ask the nraw from line 0 to 40 :" << std::endl;
	job = ext.process_from_pvrow(0, 40, 0, false);
	job->wait_end();
	std::cout << "Done. Nraw dump :" << std::endl;
	dump_nraw(ext.get_nraw());
	
	// Now read everything and dump the index table
	std::cout << "Read everything... !\n" << std::endl;
	job = ext.read_everything();
	job->wait_end();
	std::cout << "Index table :\n" << std::endl;
	PVRush::PVAggregator::list_inputs::const_iterator it;
	const PVRush::PVAggregator::list_inputs& in = ext.get_inputs();
	for (it = in.begin(); it != in.end(); it++)
		std::cout << "Source " << qPrintable((*it)->human_name()) << " has " << (*it)->last_elt_index() << " elements." << std::endl;

	// Now reask for nraw lines from 0 to 40
	std::cout << "Ask the nraw from line 0 to 40 :" << std::endl;
	job = ext.process_from_pvrow(0, 40, 0, false);
	job->wait_end();
	std::cout << "Done. Nraw dump :" << std::endl;
	dump_nraw(ext.get_nraw());

	ext.gracefully_stop_controller();	

	std::cerr << "Controller has stopped !\n" << std::endl;
}
