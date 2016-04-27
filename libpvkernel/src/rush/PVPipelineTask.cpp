/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/rush/PVPipelineTask.h>
#include <tbb/task.h>
#include <tbb/pipeline.h>
#include <tbb/tbb_exception.h>
#include <assert.h>

PVRush::PVPipelineTask::PVPipelineTask() : tbb::task(), _ntokens(240), _running(false)
{
}

void PVRush::PVPipelineTask::set_filter(tbb::filter_t<void, void> f)
{
	assert(!_running);
	_f = f;
}

void PVRush::PVPipelineTask::set_tokens(size_t tokens)
{
	_ntokens = tokens;
}

tbb::task* PVRush::PVPipelineTask::execute()
{
	assert(_ntokens > 0);
	_running = true;
	try {
#if (TBB_INTERFACE_VERSION >= 5006)
		tbb::parallel_pipeline(_ntokens, _f, *group());
#else
		tbb::parallel_pipeline(_ntokens, _f);
#endif
	} catch (tbb::captured_exception& e) {
		PVLOG_ERROR("Uncatched exception in TBB pipeline of type '%s': %s.\n", e.name(), e.what());
	}
	_running = false;
	return NULL;
}

void PVRush::PVPipelineTask::cancel()
{
	PVLOG_DEBUG("(PVPipelineTask) Pipeline asked to be canceled...\n");
	this->cancel_group_execution();
}
