#include <pvrush/PVInputFile.h>
#include <pvrush/PVChunkAlignUTF16Char.h>
#include <pvrush/PVChunkTransformUTF16.h>
#include <pvrush/PVRawSource.h>
#include <pvrush/PVOutputFile.h>
#include <pvrush/PVVrbAllocator.h>
#include <pvrush/PVController.h>
#include <pvrush/PVControllerJob.h>
#include <pvfilter/PVChunkFilter.h>
#include <pvfilter/PVElementFilterRandInvalid.h>
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

PVController _ctrl; // the controller
static void th_controller()
{
	_ctrl();
}

void chunk_filter_test(PVChunkFilterByElt &chk_flt, int argc, char** argv)
{
	if (argc < 4) {
		std::cerr << "Usage: " << argv[0] << " file chunk_size nchunks" << std::endl;
		std::exit(1);
	}

	std::thread thctrl(th_controller);

	PVInputFile ifile(argv[1]);
	PVChunkAlignUTF16Char calign(QChar('\n'));
	PVChunkTransformUTF16 transform;
	PVChunkFilter null;
	PVElementFilterRandInvalid elt_f_invalid;
	PVChunkFilterByElt chk_invalid(elt_f_invalid.f());
	const int nchunks = atoi(argv[3]);
	const int chunk_size = atoi(argv[2]);
#ifdef PV_USE_RINGBUFFER
	vrb_p v = vrb_new(nchunks*(chunk_size*sizeof(char)+100), NULL);
	PVRawSourceRB::alloc_chunk alloc(v);	
	PVRawSourceBase_p source = PVRawSourceBase_p(new PVRawSourceRB(ifile, calign, chunk_size, transform, null, alloc));
#else
	PVRawSourceBase_p source(new PVRawSource<>(ifile, calign, chunk_size, transform, null));
#endif
	PVAggregator agg = PVAggregator::from_unique_source(source);

	std::string path_out(argv[1]);
	path_out += ".out";
	PVOutputFile ofile(path_out.c_str());


	// Create a job and submit it to the controller
	PVControllerJob_p job = PVControllerJob_p(new PVControllerJob(PVControllerJob::start, 0));
	job->set_params(0, 0, 10000000, PVControllerJob::sc_n_elts, agg, chk_flt.f(), ofile, nchunks);
	//job->set_params(0, 0, 10000000, PVControllerJob::sc_n_elts, agg, boost::bind(chk_flt.f(), boost::bind(chk_invalid.f(), _1)), ofile, nchunks);
	
//	PVControllerJob job1(PVControllerJob::start, 0);
//	job1.set_params(10, 5000000, agg, chk_flt.f(), ofile, nchunks);
//
//	PVControllerJob job2(PVControllerJob::start, 0);
//	job2.set_params(0, 5000, agg, chk_flt.f(), ofile, nchunks);

	_ctrl.submit_job(job);
	std::cout << "wait_end" << std::endl;
	job->wait_end();
	std::cout << "wait_end done" << std::endl;
//	_ctrl.submit_job(job1);
//	_ctrl.submit_job(job2);

	// Wait 2s and stop it with a priority "stop" job
//	sleep(2);
//	_ctrl.submit_job(job2);
//	_ctrl.submit_job(PVControllerJob(PVControllerJob::stop_current, 1));


	_ctrl.wait_end_and_stop();
	thctrl.join();

	std::cerr << "Controller has stopped !\n" << std::endl;
}
