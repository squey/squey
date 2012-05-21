
#include <pvparallelview/PVZoomedZoneTree.h>

#define ZZT_MAX_VALUE (1 << (32-NBITS_INDEX))

PVParallelView::PVZoomedZoneTree::PVZoomedZoneTree(uint32_t max_level)
{
	uint32_t idx = 0;
	uint32_t y1_min;
	uint32_t y2_min;

	y2_min = 0;
	for(uint32_t y2 = 0; y2 < 1024; ++y2) {
		y1_min = 0;
		for(uint32_t y1 = 0; y1 < 1024; ++y1) {
			// std::cout << y1_min << " " << y1_min + (ZZT_MAX_VALUE >> 1) << " "
			//           << y2_min << " " << y2_min + (ZZT_MAX_VALUE >> 1) << std::endl;
			_trees[idx].init(y1_min, y1_min + (ZZT_MAX_VALUE >> 1),
			                 y2_min, y2_min + (ZZT_MAX_VALUE >> 1),
			                 max_level);
			y1_min += ZZT_MAX_VALUE;
		}
		y2_min += ZZT_MAX_VALUE;
	}
}

PVParallelView::PVZoomedZoneTree::~PVZoomedZoneTree()
{
}

void PVParallelView::PVZoomedZoneTree::process(const Picviz::PVPlotted::uint_plotted_table_t &plotted,
                                               PVCol col_a, PVCol col_b, PVRow nrows)
{
	uint32_t nrows_aligned = ((nrows+3)/4)*4;

	const uint32_t* pcol_a = &plotted[col_a * nrows_aligned];
	const uint32_t* pcol_b = &plotted[col_b * nrows_aligned];

	for (PVRow r = 0; r < nrows; ++r) {
		PVParallelView::PVQuadTreeEntry e;
		e.y1 = pcol_a[r];
		e.y2 = pcol_b[r];
		e.idx = r;
		_trees[compute_index(e)].insert(e);
	}
}

size_t PVParallelView::PVZoomedZoneTree::browse_tree_bci_by_y1(uint32_t y_min, uint32_t y_max,
                                                               PVHSVColor* colors, PVBCICode* codes) const
{
	uint32_t t_min = (y_min >> (32 - NBITS_INDEX)) & MASK_INT_YCOORD;
	uint32_t t_max = (y_max >> (32 - NBITS_INDEX)) & MASK_INT_YCOORD;
	uint32_t zoom = (UINT32_MAX / (y_max - y_min)) - 1;
	size_t num = 0;

	for (uint32_t i = t_min; i < t_max; ++i) {
		num += _trees[i].get_first_bci_from_y1(y_min, y_max, zoom, colors, codes + num);
	}

	return num;
}

size_t PVParallelView::PVZoomedZoneTree::browse_tree_bci_by_y2(uint32_t y_min, uint32_t y_max,
                                                               PVHSVColor* colors, PVBCICode* codes) const
{
	uint32_t t_min = 1024 * (y_min >> (32 - NBITS_INDEX)) & MASK_INT_YCOORD;
	uint32_t t_max = 1024 * (y_max >> (32 - NBITS_INDEX)) & MASK_INT_YCOORD;
	uint32_t zoom = (UINT32_MAX / (y_max - y_min)) - 1;
	size_t num = 0;

	for (uint32_t i = t_min; i < t_max; i += 1024) {
		num += _trees[i].get_first_bci_from_y2(y_min, y_max, zoom, colors, codes + num);
	}

	return num;
}
