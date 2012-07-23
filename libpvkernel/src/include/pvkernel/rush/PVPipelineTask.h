/**
 * \file PVPipelineTask.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPIPELINETASK_FILE_H
#define PVPIPELINETASK_FILE_H

#include <pvkernel/core/general.h>
#include <tbb/task.h>
#include <tbb/pipeline.h>

namespace PVRush {

class LibKernelDecl PVPipelineTask : public tbb::task {
public:
	PVPipelineTask();
	task* execute();
	void cancel();
	void set_filter(tbb::filter_t<void,void> f);
	void set_nchunks(size_t nchunks);

protected:
	tbb::filter_t<void,void> _f;
	size_t _nchunks;
	bool _running;
};


}


#endif
