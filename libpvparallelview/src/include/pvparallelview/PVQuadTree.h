/**
 * \file PVQuadTree.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PARALLELVIEW_PVQUADTREE_H
#define PARALLELVIEW_PVQUADTREE_H

#include <pvbase/types.h>

#include <bithacks.h>

#include <pvkernel/core/PVAlgorithms.h>
#include <pvkernel/core/PVHSVColor.h>
#include <pvkernel/core/PVAllocators.h>
#include <pvkernel/core/PVVector.h>
#include <pvkernel/core/picviz_bench.h>

#include <picviz/PVSelection.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVTLRBuffer.h>

/* TODO: try to move code into .cpp, etc.
 */

namespace PVParallelView {

struct extract_stat {
	static double all_dt;
	static size_t all_cnt;
	static size_t test_cnt;
	static size_t insert_cnt;
};

#define SW 0
#define SE 1
#define NW 2
#define NE 3

/*****************************************************************************
 * About Quadtrees entries
 */
#pragma pack(push)
#pragma pack(4)

struct PVQuadTreeEntry {
	uint32_t y1;
	uint32_t y2;
	PVRow    idx;

	PVQuadTreeEntry()
	{
	}

	PVQuadTreeEntry(uint32_t y1_, uint32_t y2_, PVRow r)
	{
		y1 = y1_;
		y2 = y2_;
		idx = r;
	}
};
#pragma pack(pop)

/*****************************************************************************
 * About bitfield use when extracting relevant entries from quadtrees
 */
#define __PV_IMPL_QUADTREE_BUFFER_ENTRY_COUNT (4096 * 2048)
// we store bits
#define QUADTREE_BUFFER_SIZE (__PV_IMPL_QUADTREE_BUFFER_ENTRY_COUNT >> 5)

typedef uint32_t pv_quadtree_buffer_entry_t;

/*****************************************************************************
 * About SSE use when extracting relevant entries from quadtrees
 */
//#define QUADTREE_USE_SSE_EXTRACT

static inline bool test_sse(const __m128i &sse_y1,
                            const __m128i &sse_y1_min,
                            const __m128i &sse_y1_max,
                            __m128i &sse_res)
{
	static const __m128i sse_full_ones  = _mm_set1_epi32(0xFFFFFFFF);
	static const __m128i sse_full_zeros = _mm_set1_epi32(0);

	/* expand 4x32b register into 2 2x64b registers
	 */
	const __m128i sse_y1_0 = _mm_unpacklo_epi32 (sse_y1, sse_full_zeros);
	const __m128i sse_y1_1 = _mm_unpackhi_epi32 (sse_y1, sse_full_zeros);

	/* doing registers test against min
	 */
	const __m128i sse_min0 = _mm_cmpgt_epi64(sse_y1_min, sse_y1_0);
	const __m128i sse_min1 = _mm_cmpgt_epi64(sse_y1_min, sse_y1_1);

	/* doing registers test against max
	 */
	const __m128i sse_max0 = _mm_cmpgt_epi64(sse_y1_max, sse_y1_0);
	const __m128i sse_max1 = _mm_cmpgt_epi64(sse_y1_max, sse_y1_1);

	/* results merge (by packing them from 64 bits to 32 bits)
	 *
	 * on 64 bits:
	 * res0 = [ v01 | v00 ]
	 * res1 = [ v11 | v10 ]
	 *
	 * <=>
	 *
	 * on 32 bits (because possible values are 0 or ~0)
	 * res0 = [ v01 | v01 | v00 | v00 ]
	 * res1 = [ v11 | v11 | v10 | v10 ]
	 *
	 * with a right shift of 4 bytes, we gather results in the
	 * LSB 64 bits word:
	 * =>
	 * res0 = [ 0 | v01 | v01 | v00 ]
	 * res1 = [ 0 | v11 | v11 | v10 ]
	 *                   ~~~~~~~~~~~
	 */

	__m128i sse_ms0 = _mm_srli_si128(sse_min0, 4);
	__m128i sse_ms1 = _mm_srli_si128(sse_min1, 4);

	/* a call to unpacklo_epi64 helps to gather these 64 bits
	 * word into 1 register:
	 *
	 * res = [ v11 | v10 | v01 | v00 ]
	 */
	const __m128i sse_tmin = _mm_unpacklo_epi64(sse_ms0, sse_ms1);

	/* the same for tests with max
	 */
	sse_ms0 = _mm_srli_si128(sse_max0, 4);
	sse_ms1 = _mm_srli_si128(sse_max1, 4);

	const __m128i sse_tmax = _mm_unpacklo_epi64(sse_ms0, sse_ms1);

	sse_res = _mm_andnot_si128(sse_tmin, sse_tmax);

	return _mm_testz_si128(sse_res, sse_full_ones);
}

/*****************************************************************************
 * About the quadtree
 */
// typedef PVCore::PVVector<PVQuadTreeEntry, tbb::scalable_allocator<PVQuadTreeEntry> > pvquadtree_entries_t;
// typedef PVCore::PVVector<PVQuadTreeEntry, 1000, PVCore::PVJEMallocAllocator<PVQuadTreeEntry> > pvquadtree_entries_t;
typedef PVCore::PVVector<PVQuadTreeEntry> pvquadtree_entries_t;

template<int MAX_ELEMENTS_PER_NODE = 10000, int REALLOC_ELEMENT_COUNT = 1000, int PREALLOC_ELEMENT_COUNT = 0, size_t Bbits = NBITS_INDEX>
class PVQuadTree
{
	constexpr static uint32_t mask_int_ycoord = (((uint32_t)1)<<Bbits)-1;

public:
	typedef PVTLRBuffer<Bbits> pv_tlr_buffer_t;
	typedef std::function<void(const PVQuadTreeEntry &entry, pv_tlr_buffer_t &buffer)> insert_entry_f;

public:
	PVQuadTree(uint32_t y1_min_value, uint32_t y1_max_value, uint32_t y2_min_value, uint32_t y2_max_value, int max_level)
	{
		uint32_t y1_mid = y1_min_value + ((y1_max_value - y1_min_value) >> 1);
		uint32_t y2_mid = y2_min_value + ((y2_max_value - y2_min_value) >> 1);

		init(y1_min_value, y1_mid, y2_min_value, y2_mid, max_level);
	}

	// CTOR to use with call to init()
	PVQuadTree()
	{
	}

	~PVQuadTree() {
		if (_nodes == 0) {
			_datas.clear();
		} else {
			delete [] _nodes;
		}
	}

	void init(uint32_t y1_min_value, uint32_t y1_mid_value, uint32_t y2_min_value, uint32_t y2_mid_value, int max_level)
	{
		_y1_min_value = y1_min_value;
		_y1_mid_value = y1_mid_value;
		_y2_min_value = y2_min_value;
		_y2_mid_value = y2_mid_value;
		_max_level = max_level;
		if (PREALLOC_ELEMENT_COUNT != 0) {
			_datas.reserve(PREALLOC_ELEMENT_COUNT);
		} else {
			_datas = pvquadtree_entries_t();
		}
		_nodes = 0;
	}

	void insert(const PVQuadTreeEntry &e) {
		// searching for the right child
		register PVQuadTree *qt = this;
		while (qt->_nodes != 0) {
			qt = &qt->_nodes[qt->compute_index(e)];
		}

		// insertion
		qt->_datas.push_back(e);

		// does the current node must be splitted?
		if ((qt->_datas.size() >= MAX_ELEMENTS_PER_NODE) && qt->_max_level) {
			qt->create_next_level();
		}
	}

	__attribute__((noinline)) void compact()
	{
		if (_nodes) {
			for (int i = 0; i < 4; ++i) {
				_nodes[i].compact();
			}
		} else {
			_datas.compact();
		}
	}

	inline size_t memory() const
	{
		size_t mem = sizeof (PVQuadTree) - sizeof(pvquadtree_entries_t) + _datas.memory();
		if(_nodes != 0) {
			mem += _nodes[0].memory();
			mem += _nodes[1].memory();
			mem += _nodes[2].memory();
                        mem += _nodes[3].memory();
		}
		return mem;
	}

	inline void get_first_from_y1(uint64_t y1_min, uint64_t y1_max, uint32_t zoom,
	                              uint32_t y2_count,
	                              pv_quadtree_buffer_entry_t *buffer,
	                              const insert_entry_f &insert_f,
	                              pv_tlr_buffer_t &tlr) const
	{
		visit_y1::get_n_m(*this, y1_min, y1_max, zoom, y2_count,
		                  [](const PVQuadTreeEntry &e, const uint64_t y1_min, const uint64_t y1_max) -> bool
		                  {
			                  return (e.y1 >= y1_min) && (e.y1 < y1_max);
		                  },
		                  insert_f, buffer, tlr);
	}


	inline void get_first_from_y2(uint64_t y2_min, uint64_t y2_max, uint32_t zoom,
	                              uint32_t y1_count,
	                              pv_quadtree_buffer_entry_t *buffer,
	                              const insert_entry_f &insert_f,
	                              pv_tlr_buffer_t &tlr) const
	{
		visit_y2::get_n_m(*this, y2_min, y2_max, zoom, y1_count,
		                  [](const PVQuadTreeEntry &e, const uint64_t y2_min, const uint64_t y2_max) -> bool
		                  {
			                  return (e.y2 >= y2_min) && (e.y2 < y2_max);
		                  },
		                  insert_f, buffer, tlr);
	}


	inline void get_first_sel_from_y1(uint64_t y1_min, uint64_t y1_max,
	                                  const Picviz::PVSelection &selection,
	                                  uint32_t zoom, uint32_t y2_count,
	                                  pv_quadtree_buffer_entry_t *buffer,
	                                  const insert_entry_f &insert_f,
	                                  pv_tlr_buffer_t &tlr) const
	{
		visit_y1::get_n_m(*this, y1_min, y1_max, zoom, y2_count,
		                  [&selection](const PVQuadTreeEntry &e, const uint64_t y1_min, const uint64_t y1_max) -> bool
		                  {
			                  return (e.y1 >= y1_min) && (e.y1 < y1_max)
				                  && selection.get_line(e.idx);
		                  },
		                  insert_f, buffer, tlr);
	}


	inline void get_first_sel_from_y2(uint64_t y2_min, uint64_t y2_max,
	                                  const Picviz::PVSelection &selection,
	                                  uint32_t zoom, uint32_t y1_count,
	                                  pv_quadtree_buffer_entry_t *buffer,
	                                  const insert_entry_f &insert_f,
	                                  pv_tlr_buffer_t &tlr) const
	{
		visit_y2::get_n_m(*this, y2_min, y2_max, zoom, y1_count,
		                  [&selection](const PVQuadTreeEntry &e, const uint64_t y2_min, const uint64_t y2_max) -> bool
		                  {
			                  return (e.y2 >= y2_min) && (e.y2 < y2_max)
				                  && selection.get_line(e.idx);
		                  },
		                  insert_f, buffer, tlr);
	}

	PVQuadTree *get_subtree_from_y1(uint32_t y1_min, uint32_t y1_max)
	{
		PVQuadTree *new_tree = new PVQuadTree(*this);
		new_tree->init(*this);
		get_subtree_from_y1(*new_tree, y1_min, y1_max);
		return new_tree;
	}

	PVQuadTree *get_subtree_from_y2(uint32_t y2_min, uint32_t y2_max)
	{
		PVQuadTree *new_tree = new PVQuadTree(*this);
		new_tree->init(*this);
		get_subtree_from_y2(*new_tree, y2_min, y2_max);
		return new_tree;
	}

	PVQuadTree *get_subtree_from_y1y2(uint32_t y1_min, uint32_t y1_max, uint32_t y2_min, uint32_t y2_max)
	{
		PVQuadTree *new_tree = new PVQuadTree(*this);
		new_tree->init(*this);
		get_subtree_from_y1y2(*new_tree, y1_min, y1_max, y2_min, y2_max);
		return new_tree;
	}

	PVQuadTree *get_subtree_from_selection(const Picviz::PVSelection &selection)
	{
		PVQuadTree *new_tree = new PVQuadTree(*this);
		new_tree->init(*this);
		get_subtree_from_selection(*new_tree, selection);
		return new_tree;
	}

	size_t compute_selection_y1(const uint64_t y1_min, const uint64_t y1_max, Picviz::PVSelection &selection) const
	{
		return compute_selection_y1(*this, y1_min, y1_max, selection);
	}

	size_t compute_selection_y2(const uint64_t y2_min, const uint64_t y2_max, Picviz::PVSelection &selection) const
	{
		return compute_selection_y2(*this, y2_min, y2_max, selection);
	}

	static void all_clear()
	{
		extract_stat::all_dt = 0;
	}

	static double all_get()
	{
		return extract_stat::all_dt;
	}

	static void all_count_clear()
	{
		extract_stat::all_cnt = 0;
	}

	static size_t all_count_get()
	{
		return extract_stat::all_cnt;
	}

	static void test_count_clear()
	{
		extract_stat::test_cnt = 0;
	}

	static size_t test_count_get()
	{
		return extract_stat::test_cnt;
	}

	static void insert_count_clear()
	{
		extract_stat::insert_cnt = 0;
	}

	static size_t insert_count_get()
	{
		return extract_stat::insert_cnt;
	}

private:
	void init(const PVQuadTree &qt)
	{
		_y1_min_value = qt._y1_min_value;
		_y1_mid_value = qt._y1_mid_value;
		_y2_min_value = qt._y2_min_value;
		_y2_mid_value = qt._y2_mid_value;
		_max_level = qt._max_level;
		_nodes = 0;
	}

	inline int compute_index(const PVQuadTreeEntry &e) const
	{
		return ((e.y2 > _y2_mid_value) << 1) | (e.y1 > _y1_mid_value);
	}

	void create_next_level()
	{
		uint32_t y1_step = (_y1_mid_value - _y1_min_value) >> 1;
		uint32_t y2_step = (_y2_mid_value - _y2_min_value) >> 1;

		_nodes = new PVQuadTree [4];
		_nodes[NE].init(_y1_mid_value, _y1_mid_value + y1_step,
		                _y2_mid_value, _y2_mid_value + y2_step,
		                _max_level - 1);

		_nodes[SE].init(_y1_mid_value, _y1_mid_value + y1_step,
		                _y2_min_value, _y2_min_value + y2_step,
		                _max_level - 1);

		_nodes[SW].init(_y1_min_value, _y1_min_value + y1_step,
		                _y2_min_value, _y2_min_value + y2_step,
		                _max_level - 1);

		_nodes[NW].init(_y1_min_value, _y1_min_value + y1_step,
		                _y2_mid_value, _y2_mid_value + y2_step,
		                _max_level - 1);

		for (unsigned i = 0; i < _datas.size(); ++i) {
			const PVQuadTreeEntry &e = _datas.at(i);
			_nodes[compute_index(e)]._datas.push_back(e);
		}
		_datas.clear();
	}

private:
	struct visit_y1
	{
		template <typename Ftest>
		static void get_n_m(PVQuadTree const& obj,
		                    const uint64_t y1_min, const uint64_t y1_max,
		                    const uint32_t zoom, const uint32_t y2_count,
		                    const Ftest &test_f, const insert_entry_f &insert_f,
		                    pv_quadtree_buffer_entry_t *buffer,
		                    pv_tlr_buffer_t &tlr)
		{
			if (zoom == 0) {
				/* need a 1 entry width band along y1
				 */
				get_1_m(obj, y1_min, y1_max, y2_count, test_f, insert_f, buffer, tlr);
			} else if (y2_count == 1) {
				/* need a 1 entry width band along y2
				 */
				get_n_1(obj, y1_min, y1_max, zoom, test_f, insert_f, buffer, tlr);
			} else if (obj._nodes != 0) {
				/* recursive search can be processed
				 */
				if (obj._y1_mid_value < y1_max) {
					get_n_m(obj._nodes[NE], y1_min, y1_max,
					        zoom - 1, y2_count >> 1,
					        test_f, insert_f, buffer, tlr);
					get_n_m(obj._nodes[SE], y1_min, y1_max,
					        zoom - 1, y2_count >> 1,
					        test_f, insert_f, buffer, tlr);
				}
				if (y1_min < obj._y1_mid_value) {
					get_n_m(obj._nodes[NW], y1_min, y1_max,
					        zoom - 1, y2_count >> 1,
					        test_f, insert_f, buffer, tlr);
					get_n_m(obj._nodes[SW], y1_min, y1_max,
					        zoom - 1, y2_count >> 1,
					        test_f, insert_f, buffer, tlr);
				}
			} else if (obj._datas.size() != 0) {
				/* this is a unsplitted node with data and an array of nxm
				 * entries is needed
				 */
#ifdef QUADTREE_USE_SSE_EXTRACT
				extract_sse(obj, y1_min, y1_max, zoom, y2_count,
				            test_f, insert_f, buffer, tlr);
#else
				extract_seq(obj, y1_min, y1_max, zoom, y2_count,
				            test_f, insert_f, buffer, tlr);
#endif
			}
		}

		template <typename Ftest>
		static void get_1_m(PVQuadTree const& obj,
		                    const uint64_t y1_min, const uint64_t y1_max,
		                    const uint32_t y2_count,
		                    const Ftest &test_f,
		                    const insert_entry_f &insert_f,
		                    pv_quadtree_buffer_entry_t *buffer,
		                    pv_tlr_buffer_t &tlr)
		{
			if (y2_count == 1) {
				/* time to extract
				 */
				PVQuadTreeEntry e(0, 0, UINT32_MAX);
				get_1_1(obj, y1_min, y1_max, test_f, e);
				if (e.idx != UINT32_MAX) {
					insert_f(e, tlr);
				}
			} else  if (obj._nodes != 0) {
				if (obj._y1_mid_value < y1_max) {
					get_1_m(obj._nodes[NE], y1_min, y1_max,
					        y2_count >> 1, test_f, insert_f,
					        buffer, tlr);
					get_1_m(obj._nodes[SE], y1_min, y1_max,
					        y2_count >> 1, test_f, insert_f,
					        buffer, tlr);
				}
				if (y1_min < obj._y1_mid_value) {
					get_1_m(obj._nodes[NW], y1_min, y1_max,
					        y2_count >> 1, test_f, insert_f,
					        buffer, tlr);
					get_1_m(obj._nodes[SW], y1_min, y1_max,
					        y2_count >> 1, test_f, insert_f,
					        buffer, tlr);
				}
			} else if (obj._datas.size() != 0) {
				/* this is a unsplitted node with data and an array of 1xm
				 * entries is needed
				 */
#ifdef QUADTREE_USE_SSE_EXTRACT
				extract_sse(obj, y1_min, y1_max, 0, y2_count,
				            test_f, insert_f, buffer, tlr);
#else
				extract_seq(obj, y1_min, y1_max, 0, y2_count,
				            test_f, insert_f, buffer, tlr);
#endif
			}
		}

		template <typename Ftest>
		static void get_n_1(PVQuadTree const& obj,
		                    const uint64_t y1_min, const uint64_t y1_max,
		                    const uint32_t zoom,
		                    const Ftest &test_f,
		                    const insert_entry_f &insert_f,
		                    pv_quadtree_buffer_entry_t *buffer,
		                    pv_tlr_buffer_t &tlr)
		{
			if (zoom == 0) {
				/* time to extract
				 */
				PVQuadTreeEntry e(0, 0, UINT32_MAX);
				get_1_1(obj, y1_min, y1_max, test_f, e);
				if (e.idx != UINT32_MAX) {
					insert_f(e, tlr);
				}
			} else if (obj._nodes != 0) {
				if (obj._y1_mid_value < y1_max) {
					get_n_1(obj._nodes[NE], y1_min, y1_max,
					        zoom - 1, test_f, insert_f,
					        buffer, tlr);
					get_n_1(obj._nodes[SE], y1_min, y1_max,
					        zoom - 1, test_f, insert_f,
					        buffer, tlr);
				}
				if (y1_min < obj._y1_mid_value) {
					get_n_1(obj._nodes[NW], y1_min, y1_max,
					        zoom - 1, test_f, insert_f,
					        buffer, tlr);
					get_n_1(obj._nodes[SW], y1_min, y1_max,
					        zoom - 1, test_f, insert_f,
					        buffer, tlr);
				}
			} else if (obj._datas.size() != 0) {
				/* this is a unsplitted node with data and an array of nx1
				 * entries is needed
				 */
#ifdef QUADTREE_USE_SSE_EXTRACT
				extract_sse(obj, y1_min, y1_max, zoom, 1,
				            test_f, insert_f, buffer, tlr);
#else
				extract_seq(obj, y1_min, y1_max, zoom, 1,
				            test_f, insert_f, buffer, tlr);
#endif
			}
		}

		template <typename Ftest>
		static void get_1_1(PVQuadTree const& obj,
		                    const uint64_t y1_min, const uint64_t y1_max,
		                    const Ftest &test_f, PVQuadTreeEntry &result)
		{
			if (obj._nodes != 0) {
				/* pick the first relevant element from the children
				 */
				if (obj._y1_mid_value < y1_max) {
					get_1_1(obj._nodes[NE], y1_min, y1_max, test_f, result);
					get_1_1(obj._nodes[SE], y1_min, y1_max, test_f, result);
				}
				if (y1_min < obj._y1_mid_value) {
					get_1_1(obj._nodes[NW], y1_min, y1_max, test_f, result);
					get_1_1(obj._nodes[SW], y1_min, y1_max, test_f, result);
				}
			} else {
				/* pick the first relevant element from the entry list
				 */
				for (size_t i = 0; i < obj._datas.size(); ++i) {
					const PVQuadTreeEntry &e = obj._datas.at(i);
					if (test_f(e, y1_min, y1_max) && (e.idx < result.idx)) {
						result = e;
					}
				}
			}
		}

		template <typename Ftest>
		static void extract_seq(PVQuadTree const& obj,
		                        const uint64_t y1_min, const uint64_t y1_max,
		                        const uint32_t zoom, const uint32_t y2_count,
		                        const Ftest &test_f,
		                        const insert_entry_f &insert_f,
		                        pv_quadtree_buffer_entry_t *buffer,
		                        pv_tlr_buffer_t &tlr)
		{
			BENCH_START(extract);
			const uint64_t max_count = 1 << zoom;
			const uint64_t y1_orig = obj._y1_min_value;
			const uint64_t y1_len = (obj._y1_mid_value - y1_orig) * 2;
			const uint64_t y1_scale = y1_len / max_count;

			const uint64_t y2_orig = obj._y2_min_value;
			const uint64_t y2_scale = ((obj._y2_mid_value - y2_orig) * 2) / y2_count;

			const uint64_t ly1_min = (PVCore::clamp(y1_min, y1_orig, y1_orig + y1_len) - y1_orig) / y1_scale;
			const uint64_t ly1_max = (PVCore::clamp(y1_max, y1_orig, y1_orig + y1_len) - y1_orig) / y1_scale;
			const uint64_t clipped_max_count = PVCore::max(1UL, ly1_max - ly1_min);
			const size_t count_aligned = ((clipped_max_count * y2_count) + 31) / 32;
			memset(buffer, 0, count_aligned * sizeof(uint32_t));
			uint32_t remaining = clipped_max_count * y2_count;

			for(size_t i = 0; i < obj._datas.size(); ++i) {
				++extract_stat::all_cnt;
				const PVQuadTreeEntry &e = obj._datas.at(i);
				if (!test_f(e, y1_min, y1_max)) {
					continue;
				}
				const uint32_t pos = (((e.y1 - y1_orig) / y1_scale) - ly1_min) + clipped_max_count * ((e.y2 - y2_orig) / y2_scale);
				++extract_stat::test_cnt;
				if (B_IS_SET(buffer[pos >> 5], pos & 31)) {
					continue;
				}
				insert_f(e, tlr);
				++extract_stat::insert_cnt;
				B_SET(buffer[pos >> 5], pos & 31);
				--remaining;
				if (remaining == 0) {
					break;
				}
			}
			BENCH_STOP(extract);
			extract_stat::all_dt += BENCH_END_TIME(extract);
		}

		template <typename Ftest>
		static void extract_sse(PVQuadTree const& obj,
		                        const uint64_t y1_min, const uint64_t y1_max,
		                        const uint32_t zoom, const uint32_t y2_count,
		                        const Ftest &test_f,
		                        const insert_entry_f &insert_f,
		                        pv_quadtree_buffer_entry_t *buffer,
		                        pv_tlr_buffer_t &tlr)
		{

			const uint64_t max_count = 1 << zoom;
			const uint64_t y1_orig = obj._y1_min_value;
			const uint64_t y1_len = (obj._y1_mid_value - y1_orig) * 2;
			const uint64_t y1_scale = y1_len / max_count;
			const uint64_t y1_shift = log2(y1_scale);
			const uint64_t y2_orig = obj._y2_min_value;
			const uint64_t y2_scale = ((obj._y2_mid_value - y2_orig) * 2) / y2_count;
			const uint64_t y2_shift = log2(y2_scale);
			const uint64_t ly1_min = (PVCore::clamp(y1_min, y1_orig, y1_orig + y1_len) - y1_orig) / y1_scale;
			const uint64_t ly1_max = (PVCore::clamp(y1_max, y1_orig, y1_orig + y1_len) - y1_orig) / y1_scale;
			const uint64_t clipped_max_count = PVCore::max(1UL, ly1_max - ly1_min);
			const size_t count_aligned = ((clipped_max_count * y2_count) + 31) / 32;
			memset(buffer, 0, count_aligned * sizeof(uint32_t));
			uint32_t remaining = clipped_max_count * y2_count;

			const __m128i sse_y1_min            = _mm_set1_epi64x(y1_min);
			const __m128i sse_y1_max            = _mm_set1_epi64x(y1_max);
			const __m128i sse_y1_orig           = _mm_set1_epi32(y1_orig);
			const __m128i sse_y1_shift          = _mm_set1_epi32(y1_shift);
			const __m128i sse_ly1_min           = _mm_set1_epi32(ly1_min);
			const __m128i sse_y2_orig           = _mm_set1_epi32(y2_orig);
			const __m128i sse_y2_shift          = _mm_set1_epi32(y2_shift);
			const __m128i sse_clipped_max_count = _mm_set1_epi32(clipped_max_count);

			const size_t size = obj._datas.size();
			const size_t packed_size = size & ~3;

			for(size_t i = 0; i < packed_size; i += 4) {
				const PVQuadTreeEntry &e0 = obj._datas.at(i);
				const PVQuadTreeEntry &e1 = obj._datas.at(i+1);
				const PVQuadTreeEntry &e2 = obj._datas.at(i+2);
				const PVQuadTreeEntry &e3 = obj._datas.at(i+3);

				// TODO: compact all _mm_xxxxx expressions ;-)
				__m128i sse_r0 = _mm_loadu_si128((const __m128i*) &e0);
				__m128i sse_r1 = _mm_loadu_si128((const __m128i*) &e1);
				__m128i sse_r2 = _mm_loadu_si128((const __m128i*) &e2);
				__m128i sse_r3 = _mm_loadu_si128((const __m128i*) &e3);

				/* partial "transposition" to have all y1 in one register
				 * and all y2 in an other one
				 */
				__m128i sse_tmp01 = _mm_unpacklo_epi32(sse_r0, sse_r1);
				__m128i sse_tmp23 = _mm_unpacklo_epi32(sse_r2, sse_r3);

				__m128i sse_y1 = _mm_unpacklo_epi64(sse_tmp01, sse_tmp23);

				__m128i sse_test;

				if (test_sse(sse_y1, sse_y1_min, sse_y1_max, sse_test)) {
					continue;
				}

				// sse_y2 is not needed before the call to test_sse
				__m128i sse_y2 = _mm_unpackhi_epi64(sse_tmp01, sse_tmp23);

				/*
				 * pos = (((e.y1 - y1_orig) / y1_scale) - ly1_min)
				 * + clipped_max_count * ((e.y2 - y2_orig) / y2_scale)
				 */
				__m128i sse_0s  = _mm_sub_epi32(sse_y1, sse_y1_orig);
				__m128i sse_0sd = _mm_srl_epi32(sse_0s, sse_y1_shift);
				__m128i sse_0x  = _mm_sub_epi32(sse_0sd, sse_ly1_min);

				__m128i sse_1s  = _mm_sub_epi32(sse_y2, sse_y2_orig);
				__m128i sse_1sd = _mm_srl_epi32(sse_1s, sse_y2_shift);
				__m128i sse_1y  = _mm_mullo_epi32(sse_1sd, sse_clipped_max_count);

				__m128i sse_pos = _mm_add_epi32(sse_0x, sse_1y);

				if(_mm_extract_epi32(sse_test, 0)) {
					uint32_t p = _mm_extract_epi32(sse_pos, 0);
					if (!(B_IS_SET(buffer[p >> 5], p & 31))) {
						insert_f(e0, tlr);
						B_SET(buffer[p >> 5], p & 31);
						--remaining;
						if (remaining == 0) {
							break;
						}
					}
				}

				if(_mm_extract_epi32(sse_test, 1)) {
					uint32_t p = _mm_extract_epi32(sse_pos, 1);
					if (!(B_IS_SET(buffer[p >> 5], p & 31))) {
						insert_f(e1, tlr);
						B_SET(buffer[p >> 5], p & 31);
						--remaining;
						if (remaining == 0) {
							break;
						}
					}
				}

				if(_mm_extract_epi32(sse_test, 2)) {
					uint32_t p = _mm_extract_epi32(sse_pos, 2);
					if (!(B_IS_SET(buffer[p >> 5], p & 31))) {
						insert_f(e2, tlr);
						B_SET(buffer[p >> 5], p & 31);
						--remaining;
						if (remaining == 0) {
							break;
						}
					}
				}

				if(_mm_extract_epi32(sse_test, 3)) {
					uint32_t p = _mm_extract_epi32(sse_pos, 3);
					if (!(B_IS_SET(buffer[p >> 5], p & 31))) {
						insert_f(e3, tlr);
						B_SET(buffer[p >> 5], p & 31);
						--remaining;
						if (remaining == 0) {
							break;
						}
					}
				}
			}

			for(size_t i = packed_size; i < size; ++i) {
				const PVQuadTreeEntry &e = obj._datas.at(i);
				if (!test_f(e, y1_min, y1_max)) {
					continue;
				}
				const uint32_t pos = (((e.y1 - y1_orig) / y1_scale) - ly1_min) + clipped_max_count * ((e.y2 - y2_orig) / y2_scale);
				if (B_IS_SET(buffer[pos >> 5], pos & 31)) {
					continue;
				}
				insert_f(e, tlr);
				B_SET(buffer[pos >> 5], pos & 31);
				--remaining;
				if (remaining == 0) {
					break;
				}
			}
		}
	};

	struct visit_y2
	{
		template <typename Ftest>
		static void get_n_m(PVQuadTree const& obj,
		                    const uint64_t y2_min, const uint64_t y2_max,
		                    const uint32_t zoom, const uint32_t y1_count,
		                    const Ftest &test_f,
		                    const insert_entry_f &insert_f,
		                    pv_quadtree_buffer_entry_t *buffer,
		                    pv_tlr_buffer_t &tlr)
		{
			if (zoom == 0) {
				/* need a 1 entry width band along y2
				 */
				get_1_m(obj, y2_min, y2_max, y1_count, test_f, insert_f, buffer, tlr);
			} else if (y1_count == 1) {
				/* need a 1 entry width band along y1
				 */
				get_n_1(obj, y2_min, y2_max, zoom, test_f, insert_f, buffer, tlr);
			} else if (obj._nodes != 0) {
				/* recursive search can be processed
				 */
				if (obj._y2_mid_value < y2_max) {
					get_n_m(obj._nodes[NE], y2_min, y2_max,
					        zoom - 1, y1_count >> 1,
					        test_f, insert_f, buffer, tlr);
					get_n_m(obj._nodes[NW], y2_min, y2_max,
					        zoom - 1, y1_count >> 1,
					        test_f, insert_f, buffer, tlr);
				}
				if (y2_min < obj._y2_mid_value) {
					get_n_m(obj._nodes[SE], y2_min, y2_max,
					        zoom - 1, y1_count >> 1,
					        test_f, insert_f, buffer, tlr);
					get_n_m(obj._nodes[SW], y2_min, y2_max,
					        zoom - 1, y1_count >> 1,
					        test_f, insert_f, buffer, tlr);
				}
			} else if (obj._datas.size() != 0) {
				/* this is a unsplitted node with data and an array of nxm
				 * entries is needed
				 */
#ifdef QUADTREE_USE_SSE_EXTRACT
				extract_sse(obj, y2_min, y2_max, zoom, y1_count,
				            test_f, insert_f, buffer, tlr);
#else
				extract_seq(obj, y2_min, y2_max, zoom, y1_count,
				            test_f, insert_f, buffer, tlr);
#endif
			}
		}

		template <typename Ftest>
		static void get_1_m(PVQuadTree const& obj,
		                    const uint64_t y2_min, const uint64_t y2_max,
		                    const uint32_t y1_count,
		                    const Ftest &test_f,
		                    const insert_entry_f &insert_f,
		                    pv_quadtree_buffer_entry_t *buffer,
		                    pv_tlr_buffer_t &tlr)
		{
			if (y1_count == 1) {
				/* time to extract
				 */
				PVQuadTreeEntry e(0, 0, UINT32_MAX);
				get_1_1(obj, y2_min, y2_max, test_f, e);
				if (e.idx != UINT32_MAX) {
					insert_f(e, tlr);
				}
			} else  if (obj._nodes != 0) {
				if (obj._y2_mid_value < y2_max) {
					get_1_m(obj._nodes[NE], y2_min, y2_max,
					        y1_count >> 1, test_f, insert_f,
					        buffer, tlr);
					get_1_m(obj._nodes[NW], y2_min, y2_max,
					        y1_count >> 1, test_f, insert_f,
					        buffer, tlr);
				}
				if (y2_min < obj._y2_mid_value) {
					get_1_m(obj._nodes[SE], y2_min, y2_max,
					        y1_count >> 1, test_f, insert_f,
					        buffer, tlr);
					get_1_m(obj._nodes[SW], y2_min, y2_max,
					        y1_count >> 1, test_f, insert_f,
					        buffer, tlr);
				}
			} else if (obj._datas.size() != 0) {
				/* this is a unsplitted node with data and an array of 1xm
				 * entries is needed
				 */
#ifdef QUADTREE_USE_SSE_EXTRACT
				extract_sse(obj, y2_min, y2_max, 0, y1_count,
				            test_f, insert_f, buffer, tlr);
#else
				extract_seq(obj, y2_min, y2_max, 0, y1_count,
				            test_f, insert_f, buffer, tlr);
#endif
			}
		}

		template <typename Ftest>
		static void get_n_1(PVQuadTree const& obj,
		                    const uint64_t y2_min, const uint64_t y2_max,
		                    const uint32_t zoom,
		                    const Ftest &test_f,
		                    const insert_entry_f &insert_f,
		                    pv_quadtree_buffer_entry_t *buffer,
		                    pv_tlr_buffer_t &tlr)
		{
			if (zoom == 0) {
				/* time to extract
				 */
				PVQuadTreeEntry e(0, 0, UINT32_MAX);
				get_1_1(obj, y2_min, y2_max, test_f, e);
				if (e.idx != UINT32_MAX) {
					insert_f(e, tlr);
				}
			} else if (obj._nodes != 0) {
				if (obj._y2_mid_value < y2_max) {
					get_n_1(obj._nodes[NE], y2_min, y2_max,
					        zoom - 1, test_f, insert_f, buffer, tlr);
					get_n_1(obj._nodes[NW], y2_min, y2_max,
					        zoom - 1, test_f, insert_f, buffer, tlr);
				}
				if (y2_min < obj._y2_mid_value) {
					get_n_1(obj._nodes[SE], y2_min, y2_max,
					        zoom - 1, test_f, insert_f, buffer, tlr);
					get_n_1(obj._nodes[SW], y2_min, y2_max,
					        zoom - 1, test_f, insert_f, buffer, tlr);
				}
			} else if (obj._datas.size() != 0) {
				/* this is a unsplitted node with data and an array of nx1
				 * entries is needed
				 */
#ifdef QUADTREE_USE_SSE_EXTRACT
				extract_sse(obj, y2_min, y2_max, zoom, 1,
				            test_f, insert_f, buffer, tlr);
#else
				extract_seq(obj, y2_min, y2_max, zoom, 1,
				            test_f, insert_f, buffer, tlr);
#endif
			}
		}

		template <typename Ftest>
		static void get_1_1(PVQuadTree const& obj,
		                    const uint64_t y2_min, const uint64_t y2_max,
		                    const Ftest &test_f,
		                    PVQuadTreeEntry &result)
		{
			if (obj._nodes != 0) {
				/* pick the first relevant element from the children
				 */
				if (obj._y2_mid_value < y2_max) {
					get_1_1(obj._nodes[NE], y2_min, y2_max, test_f, result);
					get_1_1(obj._nodes[SE], y2_min, y2_max, test_f, result);
				}
				if (y2_min < obj._y2_mid_value) {
					get_1_1(obj._nodes[NW], y2_min, y2_max, test_f, result);
					get_1_1(obj._nodes[SW], y2_min, y2_max, test_f, result);
				}
			} else {
				/* pick the first relevant element from the entry list
				 */
				for (size_t i = 0; i < obj._datas.size(); ++i) {
					const PVQuadTreeEntry &e = obj._datas.at(i);
					if (test_f(e, y2_min, y2_max) && (e.idx < result.idx)) {
						result = e;
					}
				}
			}
		}

		template <typename Ftest>
		static void extract_seq(PVQuadTree const& obj,
		                        const uint64_t y2_min, const uint64_t y2_max,
		                        const uint32_t zoom, const uint32_t y1_count,
		                        const Ftest &test_f,
		                        const insert_entry_f &insert_f,
		                        pv_quadtree_buffer_entry_t *buffer,
		                        pv_tlr_buffer_t &tlr)
		{
			const uint64_t max_count = 1 << zoom;
			const uint64_t y1_orig = obj._y1_min_value;
			const uint64_t y1_scale = ((obj._y1_mid_value - y1_orig) * 2) / y1_count;
			const uint64_t y2_orig = obj._y2_min_value;
			const uint64_t y2_len = (obj._y2_mid_value - y2_orig) * 2;
			const uint64_t y2_scale = y2_len / max_count;
			const uint64_t ly2_min = (PVCore::clamp(y2_min, y2_orig, y2_orig + y2_len) - y2_orig) / y2_scale;
			const uint64_t ly2_max = (PVCore::clamp(y2_max, y2_orig, y2_orig + y2_len) - y2_orig) / y2_scale;
			const uint64_t clipped_max_count = PVCore::max(1UL, ly2_max - ly2_min);
			const size_t count_aligned = ((clipped_max_count * y1_count) + 31) / 32;
			memset(buffer, 0, count_aligned * sizeof(uint32_t));
			uint32_t remaining = clipped_max_count * y1_count;
			for(size_t i = 0; i < obj._datas.size(); ++i) {
				const PVQuadTreeEntry &e = obj._datas.at(i);
				if (!test_f(e, y2_min, y2_max)) {
					continue;
				}
				const uint32_t pos = (((e.y2 - y2_orig) / y2_scale) - ly2_min) + clipped_max_count * ((e.y1 - y1_orig) / y1_scale);
				if (B_IS_SET(buffer[pos >> 5], pos & 31)) {
					continue;
				}
				insert_f(e, tlr);
				B_SET(buffer[pos >> 5], pos & 31);
				--remaining;
				if (remaining == 0) {
					break;
				}
			}
		}

		template <typename Ftest>
		static void extract_sse(PVQuadTree const& obj,
		                        const uint64_t y2_min, const uint64_t y2_max,
		                        const uint32_t zoom, const uint32_t y1_count,
		                        const Ftest &test_f,
		                        const insert_entry_f &insert_f,
		                        pv_quadtree_buffer_entry_t *buffer,
		                        pv_tlr_buffer_t &tlr)
		{
			const uint64_t max_count = 1 << zoom;
			const uint64_t y1_orig = obj._y1_min_value;
			const uint64_t y1_scale = ((obj._y1_mid_value - y1_orig) * 2) / y1_count;
			const uint64_t y1_shift = log2(y1_scale);
			const uint64_t y2_orig = obj._y2_min_value;
			const uint64_t y2_len = (obj._y2_mid_value - y2_orig) * 2;
			const uint64_t y2_scale = y2_len / max_count;
			const uint64_t y2_shift = log2(y2_scale);
			const uint64_t ly2_min = (PVCore::clamp(y2_min, y2_orig, y2_orig + y2_len) - y2_orig) / y2_scale;
			const uint64_t ly2_max = (PVCore::clamp(y2_max, y2_orig, y2_orig + y2_len) - y2_orig) / y2_scale;
			const uint64_t clipped_max_count = PVCore::max(1UL, ly2_max - ly2_min);
			const size_t count_aligned = ((clipped_max_count * y1_count) + 31) / 32;
			memset(buffer, 0, count_aligned * sizeof(uint32_t));
			uint32_t remaining = clipped_max_count * y1_count;

			const __m128i sse_y2_min            = _mm_set1_epi64x(y2_min);
			const __m128i sse_y2_max            = _mm_set1_epi64x(y2_max);
			const __m128i sse_y2_orig           = _mm_set1_epi32(y2_orig);
			const __m128i sse_y2_shift          = _mm_set1_epi32(y2_shift);
			const __m128i sse_ly2_min           = _mm_set1_epi32(ly2_min);
			const __m128i sse_y1_orig           = _mm_set1_epi32(y1_orig);
			const __m128i sse_y1_shift          = _mm_set1_epi32(y1_shift);
			const __m128i sse_clipped_max_count = _mm_set1_epi32(clipped_max_count);

			const size_t size = obj._datas.size();
			const size_t packed_size = size & ~3;

			for(size_t i = 0; i < packed_size; i += 4) {
				const PVQuadTreeEntry &e0 = obj._datas.at(i);
				const PVQuadTreeEntry &e1 = obj._datas.at(i+1);
				const PVQuadTreeEntry &e2 = obj._datas.at(i+2);
				const PVQuadTreeEntry &e3 = obj._datas.at(i+3);

				// TODO: compact all _mm_xxxxx expressions ;-)
				__m128i sse_r0 = _mm_loadu_si128((const __m128i*) &e0);
				__m128i sse_r1 = _mm_loadu_si128((const __m128i*) &e1);
				__m128i sse_r2 = _mm_loadu_si128((const __m128i*) &e2);
				__m128i sse_r3 = _mm_loadu_si128((const __m128i*) &e3);

				/* partial "transposition" to have all y1 in one register
				 * and all y2 in an other one
				 */
				__m128i sse_tmp01 = _mm_unpacklo_epi32(sse_r0, sse_r1);
				__m128i sse_tmp23 = _mm_unpacklo_epi32(sse_r2, sse_r3);

				__m128i sse_y2 = _mm_unpackhi_epi64(sse_tmp01, sse_tmp23);

				__m128i sse_test;

				if (test_sse(sse_y2, sse_y2_min, sse_y2_max, sse_test)) {
					continue;
				}

				// sse_y1 is not needed before the call to test_sse
				__m128i sse_y1 = _mm_unpacklo_epi64(sse_tmp01, sse_tmp23);

				/*
				 * pos = (((e.y2 - y2_orig) / y2_scale) - ly2_min)
				 * + clipped_max_count * ((e.y1 - y1_orig) / y1_scale);
				 */
				__m128i sse_0s  = _mm_sub_epi32(sse_y2, sse_y2_orig);
				__m128i sse_0sd = _mm_srl_epi32(sse_0s, sse_y2_shift);
				__m128i sse_0x  = _mm_sub_epi32(sse_0sd, sse_ly2_min);

				__m128i sse_1s  = _mm_sub_epi32(sse_y1, sse_y1_orig);
				__m128i sse_1sd = _mm_srl_epi32(sse_1s, sse_y1_shift);
				__m128i sse_1y  = _mm_mullo_epi32(sse_1sd, sse_clipped_max_count);

				__m128i sse_pos = _mm_add_epi32(sse_0x, sse_1y);

				if(_mm_extract_epi32(sse_test, 0)) {
					uint32_t p = _mm_extract_epi32(sse_pos, 0);
					if (!(B_IS_SET(buffer[p >> 5], p & 31))) {
						insert_f(e0, tlr);
						B_SET(buffer[p >> 5], p & 31);
						--remaining;
						if (remaining == 0) {
							break;
						}
					}
				}

				if(_mm_extract_epi32(sse_test, 1)) {
					uint32_t p = _mm_extract_epi32(sse_pos, 1);
					if (!(B_IS_SET(buffer[p >> 5], p & 31))) {
						insert_f(e1, tlr);
						B_SET(buffer[p >> 5], p & 31);
						--remaining;
						if (remaining == 0) {
							break;
						}
					}
				}

				if(_mm_extract_epi32(sse_test, 2)) {
					uint32_t p = _mm_extract_epi32(sse_pos, 2);
					if (!(B_IS_SET(buffer[p >> 5], p & 31))) {
						insert_f(e2, tlr);
						B_SET(buffer[p >> 5], p & 31);
						--remaining;
						if (remaining == 0) {
							break;
						}
					}
				}

				if(_mm_extract_epi32(sse_test, 3)) {
					uint32_t p = _mm_extract_epi32(sse_pos, 3);
					if (!(B_IS_SET(buffer[p >> 5], p & 31))) {
						insert_f(e3, tlr);
						B_SET(buffer[p >> 5], p & 31);
						--remaining;
						if (remaining == 0) {
							break;
						}
					}
				}
			}

			for(size_t i = packed_size; i < size; ++i) {
				const PVQuadTreeEntry &e = obj._datas.at(i);
				if (!test_f(e, y2_min, y2_max)) {
					continue;
				}
				const uint32_t pos = (((e.y2 - y2_orig) / y2_scale) - ly2_min) + clipped_max_count * ((e.y1 - y1_orig) / y1_scale);
				if (B_IS_SET(buffer[pos >> 5], pos & 31)) {
					continue;
				}
				insert_f(e, tlr);
				B_SET(buffer[pos >> 5], pos & 31);
				--remaining;
				if (remaining == 0) {
					break;
				}
			}
		}
	};

	void get_subtree_from_y1(PVQuadTree& new_tree, uint32_t y1_min, uint32_t y1_max)
	{
		if(_nodes != 0) {
			new_tree._nodes = new PVQuadTree [4];
			for (int i = 0; i < 4; ++i) {
				new_tree._nodes[i].init(_nodes[i]);
			}
			if(_y1_mid_value < y1_max) {
				_nodes[NE].get_subtree_from_y1(new_tree._nodes[NE], y1_min, y1_max);
				_nodes[SE].get_subtree_from_y1(new_tree._nodes[SE], y1_min, y1_max);
			}
			if(y1_min < _y1_mid_value) {
				_nodes[NW].get_subtree_from_y1(new_tree._nodes[NW], y1_min, y1_max);
				_nodes[SW].get_subtree_from_y1(new_tree._nodes[SW], y1_min, y1_max);
			}
		} else {
			new_tree._datas = _datas;
		}
	}

	void get_subtree_from_y2(PVQuadTree& new_tree, uint32_t y2_min, uint32_t y2_max)
	{
		if(_nodes != 0) {
			new_tree._nodes = new PVQuadTree [4];
			for (int i = 0; i < 4; ++i) {
				new_tree._nodes[i].init(_nodes[i]);
			}
			if(_y2_mid_value < y2_max) {
				_nodes[NW].get_subtree_from_y2(new_tree._nodes[NW], y2_min, y2_max);
				_nodes[NE].get_subtree_from_y2(new_tree._nodes[NE], y2_min, y2_max);
			}
			if(y2_min < _y2_mid_value) {
				_nodes[SW].get_subtree_from_y2(new_tree._nodes[SW], y2_min, y2_max);
				_nodes[SE].get_subtree_from_y2(new_tree._nodes[SE], y2_min, y2_max);
			}
		} else {
			new_tree._datas = _datas;
		}
	}

	void get_subtree_from_y1y2(PVQuadTree& new_tree, uint32_t y1_min, uint32_t y1_max, uint32_t y2_min, uint32_t y2_max)
	{
		if(_nodes != 0) {
			new_tree._nodes = new PVQuadTree [4];
			for (int i = 0; i < 4; ++i) {
				new_tree._nodes[i].init(_nodes[i]);
			}
			if(_y1_mid_value < y1_max) {
				if(_y2_mid_value < y2_max) {
					_nodes[NE].get_subtree_from_y1y2(new_tree._nodes[NE], y1_min, y1_max, y2_min, y2_max);
				}
				if(y2_min < _y2_mid_value) {
					_nodes[SE].get_subtree_from_y1y2(new_tree._nodes[SE], y1_min, y1_max, y2_min, y2_max);
				}
			}
			if(y1_min < _y1_mid_value) {
				if(_y2_mid_value < y2_max) {
					_nodes[NW].get_subtree_from_y1y2(new_tree._nodes[NW], y1_min, y1_max, y2_min, y2_max);
				}
				if(y2_min < _y2_mid_value) {
					_nodes[SW].get_subtree_from_y1y2(new_tree._nodes[SW], y1_min, y1_max, y2_min, y2_max);
				}
			}
		} else {
			new_tree._datas = _datas;
		}
	}

	size_t compute_selection_y1(PVQuadTree const& obj, const uint64_t y1_min, const uint64_t y1_max, Picviz::PVSelection &selection) const
	{
		size_t num = 0;
		if (obj._nodes != 0) {
			if (obj._y1_mid_value < y1_max) {
				num += compute_selection_y1(obj._nodes[NE], y1_min, y1_max, selection);
				num += compute_selection_y1(obj._nodes[SE], y1_min, y1_max, selection);
			}
			if (y1_min < obj._y1_mid_value) {
				num += compute_selection_y1(obj._nodes[NW], y1_min, y1_max, selection);
				num += compute_selection_y1(obj._nodes[SW], y1_min, y1_max, selection);
			}
		} else {
			for (size_t i = 0; i < obj._datas.size(); ++i) {
				const PVQuadTreeEntry &e = obj._datas.at(i);
				if ((y1_min <= e.y1) && (e.y1 < y1_max)) {
					selection.set_bit_fast(e.idx);
					++num;
				}
			}
		}
		return num;
	}

	size_t compute_selection_y2(PVQuadTree const& obj, const uint64_t y2_min, const uint64_t y2_max, Picviz::PVSelection &selection) const
	{
		size_t num = 0;
		if (obj._nodes != 0) {
			if (obj._y2_mid_value < y2_max) {
				num += compute_selection_y2(obj._nodes[NE], y2_min, y2_max, selection);
				num += compute_selection_y2(obj._nodes[NW], y2_min, y2_max, selection);
			}
			if (y2_min < obj._y2_mid_value) {
				num += compute_selection_y2(obj._nodes[SE], y2_min, y2_max, selection);
				num += compute_selection_y2(obj._nodes[SW], y2_min, y2_max, selection);
			}
		} else {
			for (size_t i = 0; i < obj._datas.size(); ++i) {
				const PVQuadTreeEntry &e = obj._datas.at(i);
				if ((y2_min <= e.y2) && (e.y2 < y2_max)) {
					selection.set_bit_fast(e.idx);
					++num;
				}
			}
		}
		return num;
	}

private:
	pvquadtree_entries_t  _datas;
	PVQuadTree           *_nodes;

	uint32_t              _y1_min_value;
	uint32_t              _y1_mid_value;
	uint32_t              _y2_min_value;
	uint32_t              _y2_mid_value;

	uint32_t              _max_level;
};

#undef SW
#undef SE
#undef NW
#undef NE

}

#endif // PARALLELVIEW_PVQUADTREE_H
