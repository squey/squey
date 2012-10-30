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

namespace __impl {

template <size_t Bbits=11>
class PVTLRBuffer
{
public:
	constexpr static size_t length = 3 * (1 << (2*Bbits));

public:
	struct tlr_index_t
	{
		tlr_index_t(uint32_t vv = 0U)
		{
			v = vv;
		}

		tlr_index_t(uint32_t t, uint32_t l, uint32_t r)
		{
#ifdef TLR_USE_C_BITFIELD
			v = 0U;
			s.t = t;
			s.l = l;
			s.r = r;
#else
			v = (t << (2*Bbits)) + (l << Bbits) + r;
#endif
		}

		union {
			uint32_t v;
			struct {
				uint32_t r : Bbits;
				uint32_t l : Bbits;
				uint32_t t : 2;
			} s;
		};
	};

public:
	PVTLRBuffer()
	{
		clear();
	}

	void clear()
	{
		memset(_data, -1, length * sizeof(uint32_t));
	}

	const uint32_t &operator[](size_t i) const
	{
		return _data[i];
	}

	uint32_t &operator[](size_t i)
	{
		return _data[i];
	}

private:
	uint32_t _data[length];
};

}

template <size_t Bbits>
class PVBCICode;

class PVZoomedZoneTree
{
	constexpr static size_t bbits = PARALLELVIEW_ZZT_BBITS;
	constexpr static uint32_t mask_int_ycoord = (((uint32_t)1)<<bbits)-1;

	typedef __impl::PVTLRBuffer<bbits> pv_tlr_buffer_t;
	typedef PVQuadTree<10000, 1000, 0, bbits> pvquadtree;
	typedef pvquadtree::insert_entry_f insert_entry_f;

	typedef std::function<void(const pvquadtree &tree,
	                           const uint32_t count,
	                           pv_quadtree_buffer_entry_t *buffer,
	                           const insert_entry_f &insert_f)> extract_entries_f;

public:
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

		pv_tlr_buffer_t &get_tlr_buffer()
		{
			return _tlr_buffer;
		}

		pv_quadtree_buffer_entry_t *get_quadtree_buffer()
		{
			return &_quadtree_buffer[0];
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
		pv_quadtree_buffer_entry_t _quadtree_buffer[QUADTREE_BUFFER_SIZE];
		pv_tlr_buffer_t            _tlr_buffer;
		size_t                     _index;
	};

	class context_t {
	public:
		typedef tbb::enumerable_thread_specific<zzt_tls> tls_set_t;

		context_t()
		{}

		~context_t()
		{}

		tls_set_t &get_tls() const
		{
			return _tls;
		}
	private:
		mutable tls_set_t _tls;
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

	bool is_initialized() const
	{
		return _initialized;
	}

	void reset();

	void process(const PVZoneProcessing &zp, PVZoneTree &zt);

public:
	void process_seq(const PVZoneProcessing &zp);

	__attribute__((noinline)) void process_seq_from_zt(const PVZoneProcessing &zp, PVZoneTree &zt);

	void process_omp(const PVZoneProcessing &zp);

	void process_omp_from_zt(const PVZoneProcessing &zp, PVZoneTree &zt);

	inline size_t compute_selection_y1(PVRow t1, const uint64_t y_min, const uint64_t y_max,
	                                   Picviz::PVSelection &selection) const
	{
		if (_initialized == false) {
			return 0;
		}

		size_t num = 0;
		for (uint32_t t2 = 0; t2 < 1024; ++t2) {
			PVRow tree_idx = (t2 * 1024) + t1;
			// if (_sel_elts[tree_idx] != PVROW_INVALID_VALUE) {
				num += _trees[tree_idx].compute_selection_y1(y_min, y_max, selection);
			// }
		}

		return num;
	}

	inline size_t compute_selection_y2(PVRow t2, const uint64_t y_min, const uint64_t y_max,
	                                   Picviz::PVSelection &selection) const
	{
		if (_initialized == false) {
			return 0;
		}

		size_t num = 0;
		for (uint32_t t1 = 0; t1 < 1024; ++t1) {
			PVRow tree_idx = (t2 * 1024) + t1;
			// if (_sel_elts[tree_idx] != PVROW_INVALID_VALUE) {
				num += _trees[tree_idx].compute_selection_y2(y_min, y_max, selection);
			// }
		}

		return num;
	}

	inline size_t browse_bci_by_y1(context_t &ctx,
	                               uint64_t y_min, uint64_t y_max, uint64_t y_lim,
	                               int zoom, uint32_t width,
	                               const PVCore::PVHSVColor* colors,
	                               pv_bci_code_t* codes,
	                               const float beta = 1.0f) const
	{
		if (_initialized == false) {
			return 0;
		}

#ifdef BROWSE_TBB
		return browse_trees_bci_by_y1_tbb(ctx, y_min, y_max, y_lim, zoom, width,
#else
		return browse_trees_bci_by_y1_seq(ctx, y_min, y_max, y_lim, zoom, width,
#endif
		                                  [&](const pvquadtree &tree,
		                                      const uint32_t y2_count,
		                                      pv_quadtree_buffer_entry_t *buffer,
		                                      const insert_entry_f &insert_f)
		                                  {
			                                  tree.get_first_from_y1(y_min, y_max,
			                                                         zoom, y2_count,
			                                                         buffer, insert_f);
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
		if (_initialized == false) {
			return 0;
		}

#ifdef BROWSE_TBB
		return browse_trees_bci_by_y2_tbb(ctx, y_min, y_max, y_lim, zoom, width,
#else
		return browse_trees_bci_by_y2_seq(ctx, y_min, y_max, y_lim, zoom, width,
#endif
		                                  [&](const pvquadtree &tree,
		                                      const uint32_t y1_count,
		                                      pv_quadtree_buffer_entry_t *buffer,
		                                      const insert_entry_f &insert_f)
		                                  {
			                                  tree.get_first_from_y2(y_min, y_max,
			                                                         zoom, y1_count,
			                                                         buffer, insert_f);
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
		if (_initialized == false) {
			return 0;
		}

#ifdef BROWSE_TBB
		return browse_trees_bci_by_y1_tbb(ctx, y_min, y_max, y_lim, zoom, width,
#else
		return browse_trees_bci_by_y1_seq(ctx, y_min, y_max, y_lim, zoom, width,
#endif
		                                  [&](const pvquadtree &tree,
		                                      const uint32_t y2_count,
		                                      pv_quadtree_buffer_entry_t *buffer,
		                                      const insert_entry_f &insert_f)
		                                  {
			                                  tree.get_first_sel_from_y1(y_min, y_max,
			                                                             selection,
			                                                             zoom, y2_count,
			                                                             buffer, insert_f);
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
		if (_initialized == false) {
			return 0;
		}

#ifdef BROWSE_TBB
		return browse_trees_bci_by_y2_tbb(ctx, y_min, y_max, y_lim, zoom, width,
#else
		return browse_trees_bci_by_y2_seq(ctx, y_min, y_max, y_lim, zoom, width,
#endif
		                                  [&](const pvquadtree &tree,
		                                      const uint32_t y1_count,
		                                      pv_quadtree_buffer_entry_t *buffer,
		                                      const insert_entry_f &insert_f)
		                                  {
			                                  tree.get_first_sel_from_y2(y_min, y_max,
			                                                             selection,
			                                                             zoom, y1_count,
			                                                             buffer, insert_f);
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
		if (_initialized == false) {
			PVLOG_WARN("you forget to initialize the ZoomedZoneTree\n");
			return 0;
		}

		return browse_trees_bci_by_y1_seq(ctx, y_min, y_max, y_lim, zoom, width,
		                                  [&](const pvquadtree &tree,
		                                      const uint32_t y2_count,
		                                      pv_quadtree_buffer_entry_t *buffer,
		                                      const insert_entry_f &insert_f)
		                                  {
			                                  tree.get_first_from_y1(y_min, y_max,
			                                                         zoom, y2_count,
			                                                         buffer, insert_f);
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
		if (_initialized == false) {
			PVLOG_WARN("you forget to initialize the ZoomedZoneTree\n");
			return 0;
		}

		return browse_trees_bci_by_y1_tbb(ctx, y_min, y_max, y_lim, zoom, width,
		                                  [&](const pvquadtree &tree,
		                                      const uint32_t y2_count,
		                                      pv_quadtree_buffer_entry_t *buffer,
		                                      const insert_entry_f &insert_f)
		                                  {
			                                  tree.get_first_from_y1(y_min, y_max,
			                                                         zoom, y2_count,
			                                                         buffer, insert_f);
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
		if (_initialized == false) {
			PVLOG_WARN("you forget to initialize the ZoomedZoneTree\n");
			return 0;
		}

		return browse_trees_bci_by_y2_seq(ctx, y_min, y_max, y_lim, zoom, width,
		                                  [&](const pvquadtree &tree,
		                                      const uint32_t y1_count,
		                                      pv_quadtree_buffer_entry_t *buffer,
		                                      const insert_entry_f &insert_f)
		                                  {
			                                  tree.get_first_from_y2(y_min, y_max,
			                                                         zoom, y1_count,
			                                                         buffer, insert_f);
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
		if (_initialized == false) {
			PVLOG_WARN("you forget to initialize the ZoomedZoneTree\n");
			return 0;
		}

		return browse_trees_bci_by_y2_tbb(ctx, y_min, y_max, y_lim, zoom, width,
		                                  [&](const pvquadtree &tree,
		                                      const uint32_t y1_count,
		                                      pv_quadtree_buffer_entry_t *buffer,
		                                      const insert_entry_f &insert_f)
		                                  {
			                                  tree.get_first_from_y2(y_min, y_max,
			                                                         zoom, y1_count,
			                                                         buffer, insert_f);
		                                  },
		                                  colors, codes, beta);
	}

private:
	size_t browse_trees_bci_by_y1_seq(context_t &ctx,
	                                  uint64_t y_min, uint64_t y_max, uint64_t y_lim, int zoom,
	                                  uint32_t width,
	                                  const extract_entries_f &extract_f,
	                                  const PVCore::PVHSVColor *colors, pv_bci_code_t *codes,
	                                  const float beta = 1.0f,
	                                  const bool use_sel = false) const;

	size_t browse_trees_bci_by_y2_seq(context_t &ctx,
	                                  uint64_t y_min, uint64_t y_max, uint64_t y_lim, int zoom,
	                                  uint32_t width,
	                                  const extract_entries_f &extract_f,
	                                  const PVCore::PVHSVColor *colors, pv_bci_code_t *codes,
	                                  const float beta = 1.0f,
	                                  const bool use_sel = false) const;

	size_t browse_trees_bci_by_y1_tbb(context_t &ctx,
	                                  uint64_t y_min, uint64_t y_max, uint64_t y_lim, int zoom,
	                                  uint32_t width,
	                                  const extract_entries_f &extract_f,
	                                  const PVCore::PVHSVColor *colors, pv_bci_code_t *codes,
	                                  const float beta = 1.0f,
	                                  const bool use_sel = false) const;

	size_t browse_trees_bci_by_y2_tbb(context_t &ctx,
	                                  uint64_t y_min, uint64_t y_max, uint64_t y_lim, int zoom,
	                                  uint32_t width,
	                                  const extract_entries_f &extract_f,
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
	uint32_t         _max_level;
	bool             _initialized;
};

typedef boost::shared_ptr<PVZoomedZoneTree> PVZoomedZoneTree_p;

}

#endif //  PARALLELVIEW_PVZOOMEDZONETREE_H
