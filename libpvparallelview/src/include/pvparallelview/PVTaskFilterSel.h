/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPARALLELVIEW_PVTASKFILTERSEL_H
#define PVPARALLELVIEW_PVTASKFILTERSEL_H

#include <tbb/task.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVZonesManager.h>

namespace Inendi {
class PVSelection;
}

namespace PVParallelView {

class PVTaskFilterSel: public tbb::task
{
public:
	PVTaskFilterSel(PVZonesManager& zm, PVZoneID z, Inendi::PVSelection const& sel):
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
	Inendi::PVSelection const& _sel;
	PVZoneID _z;
};

}

#endif
