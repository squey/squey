#ifndef PVPIPELINETASK_FILE_H
#define PVPIPELINETASK_FILE_H

#include <pvcore/general.h>
#include <tbb/task.h>
#include <tbb/pipeline.h>

namespace PVRush {

class LibRushDecl PVPipelineTask : public tbb::task {
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
