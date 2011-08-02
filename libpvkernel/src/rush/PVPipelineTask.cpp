#include <pvkernel/rush/PVPipelineTask.h>
#include <tbb/task.h>
#include <tbb/pipeline.h>
#include <assert.h>

PVRush::PVPipelineTask::PVPipelineTask() :
	tbb::task(), _nchunks(0), _running(false)
{
}



void PVRush::PVPipelineTask::set_filter(tbb::filter_t<void,void> f)
{
	assert(!_running);
	_f = f;
}

void PVRush::PVPipelineTask::set_nchunks(size_t nchunks)
{
	_nchunks = nchunks;
}

tbb::task* PVRush::PVPipelineTask::execute()
{
	assert(_nchunks > 0);
	_running = true;
#if (TBB_INTERFACE_VERSION >= 5006)
	tbb::parallel_pipeline(_nchunks, _f, *group());
#else
	tbb::parallel_pipeline(_nchunks, _f);
#endif
	_running = false;
	return NULL;
}

void PVRush::PVPipelineTask::cancel()
{
	PVLOG_DEBUG("(PVPipelineTask) Pipeline asked to be canceled...\n");
	this->cancel_group_execution();
}
