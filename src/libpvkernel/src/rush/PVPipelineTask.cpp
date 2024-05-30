//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

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
