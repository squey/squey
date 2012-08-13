/**
 * \file PVZoomedZoneTree.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PARALLELVIEW_PVZOOMEDZONETREE_H
#define PARALLELVIEW_PVZOOMEDZONETREE_H

#include <pvbase/types.h>
#include <pvparallelview/common.h>
#include <picviz/PVPlotted.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVQuadTree.h>
#include <pvparallelview/PVZoneProcessing.h>
#include <pvparallelview/PVZoneTree.h>

#include <boost/shared_ptr.hpp>

#include <tbb/tick_count.h>

namespace PVParallelView {

template <size_t Bbits>
class PVBCICode;

class PVHSVColor;

class PVZoomedZoneTree
{
	constexpr static size_t bbits = PARALLELVIEW_ZZT_BBITS;
	constexpr static uint32_t mask_int_ycoord = (((uint32_t)1)<<bbits)-1;

	typedef PVQuadTree<10000, 1000, 0, bbits> pvquadtree;

public:
	PVZoomedZoneTree(uint32_t max_level = 8);

	~PVZoomedZoneTree();

	inline size_t memory() const
	{
		size_t mem = sizeof(PVZoomedZoneTree);
		for (uint32_t i = 0; i < NBUCKETS; ++i) {
			mem += _trees[i].memory();
		}
		return mem;
	}

	inline void process(const PVZoneProcessing &zp, PVZoneTree &zt)
	{
		tbb::tick_count start, end;
		start = tbb::tick_count::now();
		process_omp_from_zt(zp, zt);
		end = tbb::tick_count::now();
		PVLOG_INFO("PVZoomedZoneTree::process in %0.4f ms.\n", (end-start).seconds()*1000.0);
		PVLOG_INFO("PVZoomedZoneTree::memory: %lu octets.\n", memory());
	}

public:
	void process_seq(const PVZoneProcessing &zp);

	void process_seq_from_zt(const PVZoneProcessing &zp, PVZoneTree &zt);

	void process_omp(const PVZoneProcessing &zp);

	void process_omp_from_zt(const PVZoneProcessing &zp, PVZoneTree &zt);

	size_t browse_tree_bci_by_y1(uint32_t y_min, int zoom,
	                             const PVHSVColor* colors, PVBCICode<bbits>* codes,
	                             const float beta = 1.0f) const;

	size_t browse_tree_bci_by_y2(PVRow y_min, int zoom,
	                             const PVHSVColor* colors, PVBCICode<bbits>* codes,
	                             const float beta = 1.0f) const;

	size_t browse_tree_bci_by_y1_range(uint32_t y_min, uint32_t y_max, int zoom,
	                                   const PVHSVColor* colors, PVBCICode<bbits>* codes,
	                                   const float beta = 1.0f) const;

private:
	inline uint32_t compute_index(uint32_t y1, uint32_t y2) const
	{
		return  (((y2 >> (32-NBITS_INDEX)) & MASK_INT_YCOORD) << NBITS_INDEX) +
			((y1 >> (32-NBITS_INDEX)) & MASK_INT_YCOORD);
	}

	inline uint32_t compute_index(const PVParallelView::PVQuadTreeEntry &e) const
	{
		return compute_index(e.y1, e.y2);
	}

private:
	pvquadtree *_trees;
	PVQuadTreeEntry *_quad_entries;
};

typedef boost::shared_ptr<PVZoomedZoneTree> PVZoomedZoneTree_p;

}

#endif //  PARALLELVIEW_PVZOOMEDZONETREE_H
