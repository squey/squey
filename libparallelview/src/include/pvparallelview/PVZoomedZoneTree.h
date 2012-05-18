#ifndef PARALLELVIEW_PVZOOMEDZONETREE_H
#define PARALLELVIEW_PVZOOMEDZONETREE_H

#include <pvbase/types.h>

#include <picviz/PVPlotted.h>

#include <pvparallelview/PVQuadTree.h>

namespace PVParallelView {

class PVZoomedZoneTree
{
public:
	PVZoomedZoneTree();

	~PVZoomedZoneTree();

	void process(const Picviz::PVPlotted &plotted, PVCol axe1, PVCol axe2, int num_lines);


private:
	PVQuadTree<22> _trees [1024 * 1024];
};

}
#endif //  PARALLELVIEW_PVZOOMEDZONETREE_H
