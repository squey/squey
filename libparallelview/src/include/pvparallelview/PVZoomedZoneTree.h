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

#include <functional>

#include <tbb/tick_count.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>

#define BROWSE_TBB

namespace PVCore {
class PVHSVColor;
}

namespace PVParallelView {

template <size_t Bbits>
class PVBCICode;

class PVZoomedZoneTree
{
	constexpr static size_t bbits = PARALLELVIEW_ZZT_BBITS;
	constexpr static uint32_t mask_int_ycoord = (((uint32_t)1)<<bbits)-1;

	typedef PVQuadTree<10000, 1000, 0, bbits> pvquadtree;
	typedef std::function<size_t(const pvquadtree &tree, PVQuadTreeEntry *entries)> extract_entry_f;
	typedef PVBCICode<bbits> pv_bci_code_t;

public:
	class zzt_tls
	{
	public:
		zzt_tls()
		{
			_index = 0;
		}

		~zzt_tls()
		{
		}

		PVQuadTreeEntry *get_quadtree_entries()
		{
			return &_quadtree_entries[0];
		}

		pv_bci_code_t *get_bci_codes()
		{
			return &_bci_codes[0];
		}

		size_t get_index() const
		{
			return _index;
		}

		void set_index(const size_t index)
		{
			_index = index;
		}

	private:
		// TODO: are the arrays correctly sized?
		PVQuadTreeEntry _quadtree_entries[NBUCKETS];
		pv_bci_code_t   _bci_codes[NBUCKETS];
		size_t          _index;
	};

	class context_t {
	public:
		typedef tbb::enumerable_thread_specific<zzt_tls> tls_set_t;

		context_t()
		{
			// TODO: is _quadtree_entries correctly sized?
			_quadtree_entries = new PVQuadTreeEntry [NBUCKETS];
		}

		~context_t()
		{
			delete [] _quadtree_entries;
		}

		PVQuadTreeEntry *get_quadtree_entries() const
		{
			return _quadtree_entries;
		}

		tls_set_t &get_tls() const
		{
			return _tls;
		}
	private:
		PVQuadTreeEntry   *_quadtree_entries;
		mutable tls_set_t  _tls;
	};

public:
	PVZoomedZoneTree(const PVRow *sel_elts, uint32_t max_level = 8);

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

	inline size_t browse_bci_by_y1(context_t &ctx,
	                               uint64_t y_min, uint64_t y_max, uint64_t y_lim,
	                               int zoom, uint32_t width,
	                               const PVCore::PVHSVColor* colors,
	                               pv_bci_code_t* codes,
	                               const float beta = 1.0f) const
	{
#ifdef BROWSE_TBB
		return browse_trees_bci_by_y1_tbb(ctx, y_min, y_max, y_lim, zoom, width,
#else
		return browse_trees_bci_by_y1_seq(ctx, y_min, y_max, y_lim, zoom, width,
#endif
		                                  [&](const pvquadtree &tree,
		                                      PVQuadTreeEntry* entries) -> size_t
		                                  {
			                                  return tree.get_first_from_y1(y_min, y_max,
			                                                                zoom, entries);
		                                  },
		                                  colors, codes, beta);
	}

	inline size_t browse_bci_by_y2(context_t &ctx,
	                               uint64_t y_min, uint64_t y_max, uint64_t y_lim,
	                               int zoom, uint32_t width,
	                               const PVCore::PVHSVColor *colors,
	                               pv_bci_code_t *codes,
	                               const float beta = 1.0f) const
	{
#ifdef BROWSE_TBB
		return browse_trees_bci_by_y2_tbb(ctx, y_min, y_max, y_lim, zoom, width,
#else
		return browse_trees_bci_by_y2_seq(ctx, y_min, y_max, y_lim, zoom, width,
#endif
		                                  [&](const pvquadtree &tree,
		                                      PVQuadTreeEntry* entries) -> size_t
		                                  {
			                                  return tree.get_first_from_y2(y_min, y_max,
			                                                                zoom, entries);
		                                  },
		                                  colors, codes, beta);
	}

	inline size_t browse_bci_sel_by_y1(context_t &ctx,
	                                   uint64_t y_min, uint64_t y_max, uint64_t y_lim,
	                                   const Picviz::PVSelection &selection,
	                                   int zoom, uint32_t width,
	                                   const PVCore::PVHSVColor *colors,
	                                   pv_bci_code_t *codes,
	                                   const float beta = 1.0f) const
	{
#ifdef BROWSE_TBB
		return browse_trees_bci_by_y1_tbb(ctx, y_min, y_max, y_lim, zoom, width,
#else
		return browse_trees_bci_by_y1_seq(ctx, y_min, y_max, y_lim, zoom, width,
#endif
		                                  [&](const pvquadtree &tree,
		                                      PVQuadTreeEntry* entries) -> size_t
		                                  {
			                                  return tree.get_first_sel_from_y1(y_min, y_max,
			                                                                    selection,
			                                                                    zoom, entries);
		                                  },
		                                  colors, codes, beta, true);
	}

	inline size_t browse_bci_sel_by_y2(context_t &ctx,
	                                   uint64_t y_min, uint64_t y_max, uint64_t y_lim,
	                                   const Picviz::PVSelection &selection,
	                                   int zoom, uint32_t width,
	                                   const PVCore::PVHSVColor *colors,
	                                   pv_bci_code_t *codes,
	                                   const float beta = 1.0f) const
	{
#ifdef BROWSE_TBB
		return browse_trees_bci_by_y2_tbb(ctx, y_min, y_max, y_lim, zoom, width,
#else
		return browse_trees_bci_by_y2_seq(ctx, y_min, y_max, y_lim, zoom, width,
#endif
		                                  [&](const pvquadtree &tree,
		                                      PVQuadTreeEntry* entries) -> size_t
		                                  {
			                                  return tree.get_first_sel_from_y2(y_min, y_max,
			                                                                    selection,
			                                                                    zoom, entries);
		                                  },
		                                  colors, codes, beta, true);
	}

	// needed for test program quadtree_browse
	inline size_t browse_bci_by_y1_seq(context_t &ctx,
	                                   uint64_t y_min, uint64_t y_max, uint64_t y_lim,
	                                   int zoom, uint32_t width,
	                                   const PVCore::PVHSVColor *colors,
	                                   pv_bci_code_t *codes,
	                                   const float beta = 1.0f) const
	{
		return browse_trees_bci_by_y1_seq(ctx, y_min, y_max, y_lim, zoom, width,
		                                  [&](const pvquadtree &tree,
		                                      PVQuadTreeEntry* entries) -> size_t
		                                  {
			                                  return tree.get_first_from_y1(y_min, y_max,
			                                                                zoom, entries);
		                                  },
		                                  colors, codes, beta);
	}

	inline size_t browse_bci_by_y1_tbb(context_t &ctx,
	                                   uint64_t y_min, uint64_t y_max, uint64_t y_lim,
	                                   int zoom, uint32_t width,
	                                   const PVCore::PVHSVColor *colors,
	                                   pv_bci_code_t *codes,
	                                   const float beta = 1.0f) const
	{
		return browse_trees_bci_by_y1_tbb(ctx, y_min, y_max, y_lim, zoom, width,
		                                  [&](const pvquadtree &tree,
		                                      PVQuadTreeEntry* entries) -> size_t
		                                  {
			                                  return tree.get_first_from_y1(y_min, y_max,
			                                                                zoom, entries);
		                                  },
		                                  colors, codes, beta);
	}

	inline size_t browse_bci_by_y2_seq(context_t &ctx,
	                                   uint64_t y_min, uint64_t y_max, uint64_t y_lim,
	                                   int zoom, uint32_t width,
	                                   const PVCore::PVHSVColor *colors,
	                                   pv_bci_code_t *codes,
	                                   const float beta = 1.0f) const
	{
		return browse_trees_bci_by_y2_seq(ctx, y_min, y_max, y_lim, zoom, width,
		                                  [&](const pvquadtree &tree,
		                                      PVQuadTreeEntry* entries) -> size_t
		                                  {
			                                  return tree.get_first_from_y2(y_min, y_max,
			                                                                zoom, entries);
		                                  },
		                                  colors, codes, beta);
	}

	inline size_t browse_bci_by_y2_tbb(context_t &ctx,
	                                   uint64_t y_min, uint64_t y_max, uint64_t y_lim,
	                                   int zoom, uint32_t width,
	                                   const PVCore::PVHSVColor *colors,
	                                   pv_bci_code_t *codes,
	                                   const float beta = 1.0f) const
	{
		return browse_trees_bci_by_y2_tbb(ctx, y_min, y_max, y_lim, zoom, width,
		                                  [&](const pvquadtree &tree,
		                                      PVQuadTreeEntry* entries) -> size_t
		                                  {
			                                  return tree.get_first_from_y2(y_min, y_max,
			                                                                zoom, entries);
		                                  },
		                                  colors, codes, beta);
	}

private:
	size_t browse_trees_bci_by_y1_seq(context_t &ctx,
	                                  uint64_t y_min, uint64_t y_max, uint64_t y_lim, int zoom,
	                                  uint32_t width,
	                                  const extract_entry_f &extract_entry,
	                                  const PVCore::PVHSVColor *colors, pv_bci_code_t *codes,
	                                  const float beta = 1.0f,
	                                  const bool use_sel = false) const;

	size_t browse_trees_bci_by_y2_seq(context_t &ctx,
	                                  uint64_t y_min, uint64_t y_max, uint64_t y_lim, int zoom,
	                                  uint32_t width,
	                                  const extract_entry_f &extract_entry,
	                                  const PVCore::PVHSVColor *colors, pv_bci_code_t *codes,
	                                  const float beta = 1.0f,
	                                  const bool use_sel = false) const;

	size_t browse_trees_bci_by_y1_tbb(context_t &ctx,
	                                  uint64_t y_min, uint64_t y_max, uint64_t y_lim, int zoom,
	                                  uint32_t width,
	                                  const extract_entry_f &extract_entry,
	                                  const PVCore::PVHSVColor *colors, pv_bci_code_t *codes,
	                                  const float beta = 1.0f,
	                                  const bool use_sel = false) const;

	size_t browse_trees_bci_by_y2_tbb(context_t &ctx,
	                                  uint64_t y_min, uint64_t y_max, uint64_t y_lim, int zoom,
	                                  uint32_t width,
	                                  const extract_entry_f &extract_entry,
	                                  const PVCore::PVHSVColor *colors, pv_bci_code_t *codes,
	                                  const float beta = 1.0f,
	                                  const bool use_sel = false) const;

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
	pvquadtree      *_trees;
	const PVRow     *_sel_elts;
};

typedef boost::shared_ptr<PVZoomedZoneTree> PVZoomedZoneTree_p;

}

#endif //  PARALLELVIEW_PVZOOMEDZONETREE_H
