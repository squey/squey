#ifndef PARALLELVIEW_PVZOOMEDZONETREE_H
#define PARALLELVIEW_PVZOOMEDZONETREE_H

#include <pvbase/types.h>
#include <pvparallelview/common.h>

#include <picviz/PVPlotted.h>

#include <pvparallelview/PVQuadTree.h>

namespace PVParallelView {

class PVBCICode;
class PVHSVColor;

class PVZoomedZoneTree
{
public:
	PVZoomedZoneTree(uint32_t max_level);

	~PVZoomedZoneTree();

	inline size_t memory() const
	{
		size_t mem = 0;
		for (int i = 0; i < 1024 * 1024; ++i) {
			mem += _trees[i].memory();
		}
		return mem;
	}

	void process(const Picviz::PVPlotted::uint_plotted_table_t &plotted,
	             PVCol col_a, PVCol col_b, PVRow nrows);

	size_t browse_tree_bci_by_y1(uint32_t y_min, uint32_t y_max,
	                             PVHSVColor* colors, PVBCICode* codes) const;

	size_t browse_tree_bci_by_y2(uint32_t y_min, uint32_t y_max,
	                             PVHSVColor* colors, PVBCICode* codes) const;

private:
	inline uint32_t compute_index(const PVParallelView::PVQuadTreeEntry &e) const
	{
		return  (((e.y2 >> (32-NBITS_INDEX)) & MASK_INT_YCOORD) << NBITS_INDEX) +
			((e.y1 >> (32-NBITS_INDEX)) & MASK_INT_YCOORD);
	}

private:
	PVQuadTree _trees [1024 * 1024];
};

}
#endif //  PARALLELVIEW_PVZOOMEDZONETREE_H
