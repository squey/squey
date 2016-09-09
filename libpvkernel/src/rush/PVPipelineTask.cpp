/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/rush/PVPipelineTask.h> // for PVPipelineTask

#include <pvkernel/core/PVLogger.h> // for PVLOG_DEBUG

#include <tbb/pipeline.h> // for filter_t, parallel_pipeline
#include <tbb/task.h>     // for task

#include <cassert> // for assert
#include <cstddef> // for size_t

PVRush::PVPipelineTask::PVPipelineTask() : tbb::task(), _ntokens(240)
{
}

void PVRush::PVPipelineTask::set_filter(tbb::filter_t<void, void> f)
{
	_f = f;
}

void PVRush::PVPipelineTask::set_tokens(size_t tokens)
{
	_ntokens = tokens;
}

tbb::task* PVRush::PVPipelineTask::execute()
{
	assert(_ntokens > 0);
	tbb::parallel_pipeline(_ntokens, _f, *group());
	return nullptr;
}

void PVRush::PVPipelineTask::cancel()
{
	PVLOG_DEBUG("(PVPipelineTask) Pipeline asked to be canceled...\n");
	this->cancel_group_execution();
}
