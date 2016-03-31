/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPIPELINETASK_FILE_H
#define PVPIPELINETASK_FILE_H

#include <pvkernel/core/general.h>
#include <tbb/task.h>
#include <tbb/pipeline.h>

namespace PVRush {

class PVPipelineTask : public tbb::task {
public:
	PVPipelineTask();
	task* execute();
	void cancel();
	void set_filter(tbb::filter_t<void,void> f);
	void set_tokens(size_t tokens);

protected:
	tbb::filter_t<void,void> _f;
	size_t _ntokens; //!< Number of tokens use in the TBB Pipeline
	bool _running;
};


}


#endif
