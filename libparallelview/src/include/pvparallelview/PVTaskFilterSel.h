#ifndef PVPARALLELVIEW_PVTASKFILTERSEL_H
#define PVPARALLELVIEW_PVTASKFILTERSEL_H

#include <tbb/task.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVZonesManager.h>

namespace Picviz {
class PVSelection;
}

namespace PVParallelView {

class PVTaskFilterSel: public tbb::task
{
public:
	PVTaskFilterSel(PVZonesManager& zm, PVZoneID z, Picviz::PVSelection const& sel):
		_zm(zm),
		_sel(sel),
		_z(z)
	{ }

public:
	virtual tbb::task* execute() override
	{
		_zm.filter_zone_by_sel(_z, _sel);
		return NULL;
	}

private:
	PVZonesManager& _zm;
	Picviz::PVSelection const& _sel;
	PVZoneID _z;
};

}

#endif
