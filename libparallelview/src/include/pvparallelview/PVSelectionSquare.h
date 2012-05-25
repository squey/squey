#ifndef PVSELECTIONSQUARE_H_
#define PVSELECTIONSQUARE_H_

#include <picviz/PVSelection.h>
#include <pvparallelview/PVZoneTree.h>
#include <pvparallelview/PVZonesManager.h>

namespace PVParallelView
{

class PVSelectionSquare
{
public:
	PVSelectionSquare(PVZonesManager& zm) : _zm(zm) {};

	void compute_selection(PVZoneID zid, QRect rect)
	{
		const PVZoneTree& ztree = _zm.get_zone_tree<PVZoneTree>(zid);
	}

	PVZonesManager& _zm;
};

}

#endif /* PVSELECTIONSQUARE_H_ */
