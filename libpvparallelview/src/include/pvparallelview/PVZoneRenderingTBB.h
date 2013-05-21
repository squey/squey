#ifndef PVPARALLELVIEW_PVZONERENDERINGTBB_H
#define PVPARALLELVIEW_PVZONERENDERINGTBB_H

#include <pvparallelview/PVZoneRendering.h>

#include <tbb/task.h>

namespace PVParallelView {

class PVZoneRenderingTBB: public PVZoneRenderingBase
{
public:
	PVZoneRenderingTBB(PVZoneID zone_id):
		PVZoneRenderingBase(zone_id)
	{ }

	PVZoneRenderingTBB():
		PVZoneRenderingBase()
	{ }

public:
	bool cancel() override
	{
		const bool ret = PVZoneRenderingBase::cancel();
		if (!ret) {
			_grp_ctxt.cancel_group_execution();
		}
		return ret;
	}

	tbb::task_group_context& get_task_group_context() { return _grp_ctxt; }

private:
	tbb::task_group_context _grp_ctxt;
};

}

#endif
