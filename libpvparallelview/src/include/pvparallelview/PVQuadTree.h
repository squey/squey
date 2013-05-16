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
#include <pvkernel/core/PVLogger.h>

#include <picviz/PVSelection.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVTLRBuffer.h>

#include <stdio.h>
#include <string.h>
#include <errno.h>

/* TODO: try to move code into .cpp, etc.
 *
 * TODO: replace the visitors zoom parameter with a y?_count
 */

namespace PVParallelView {

#ifdef PICVIZ_DEVELOPER_MODE
struct extract_stat {
	static double all_dt;
	static size_t all_cnt;
	static size_t test_cnt;
	static size_t insert_cnt;
};
#endif

#define SW 0
#define SE 1
#define NW 2
#define NE 3

/*****************************************************************************
 * About Quadtrees entries
 */
#pragma pack(push)
#pragma pack(4)

/**
 * @struct PVQuadTreeEntry
 *
 * This structure is the quadtree's internal structure used to store events
 */
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

	/* results merge (by unpacking them from 64 bits to 32 bits)
	 *
	 * on 64 bits:
	 * res0 = [ v01 | v00 ]
	 * res1 = [ v11 | v10 ]
	 *
	 * <=>
	 *
	 * on 32 bits (because lone possible values are 0 or ~0)
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
	 * words into 1 register:
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

/**
 * @class PVQuadTree
 *
 * This class implements a quadtree to store events.
 *
 * It has been initially written to have an efficient data structure when
 * zooming in views based on parallel coordinates.
 *
 * This implementation does not use usual range bounds to describe the
 * quadtree's bounding rectangle. The upper bound is replaced with the
 * range's middle because this value is more often used to traverse a
 * quadtree than the upper bound (which can computed from lower bound
 * and the middle).
 *
 * The extraction processes are done using 2 types of visitors: one when
 * constraints are on the y1 coordinate which correspond to the left border of
 * a visual representation of events, the other when constraints are on the y2
 * coordinate which correspond to the right border of a visual representation
 * of events. In fact, there is constraints on the other coordinate too.
 * Visitors also traverse the quadtree using strong constraints on one
 * coordinate and weak constraints on the other coordinate.
 *
 * weak constraints are a number of elements uniformly distributed along the
 * coordinate. Strong constraints are a half closed range and a zoom level; the
 * zoom level is internally used to compute the number of uniformly distributed
 * elements along the coordinate. Uniform distribution is discussed later.
 *
 * As we store in the quadtree only the Y coordinate of 2D points pairs
 * defining lines between 2 axes, the stored datas in the quadtree can be
 * considered as 2D points. The used extraction process has been adapted to
 * take it into account.
 *
 * A little mental sketch will be helpful :-)
 *
 * Consider a quadtree whose coordinates ranges are equal; i.e. its bounding
 * quadrilateral in an orthogonal cartesian coordinate system is a rectangle.
 * If we want to render this quadtree as a 2 pixels height image, the number
 * of lines to extract is
 *  easy to compute, it's 4:
 * - pixel 0 to pixel 0
 * - pixel 0 to pixel 1
 * - pixel 1 to pixel 0
 * - pixel 1 to pixel 1
 *
 * In quadtree's space, these resulting lines can be considered (respectively)
 * as:
 * - one point in the bottom left quarter
 * - one point in the top left quarter
 * - one point in the bottom right quarter
 * - one point in the top right quarter
 *
 * So that, extracting lines to fill a N pixels height image correspond to
 * extracting points which are uniformly distributed on a NxN grid.
 *
 * As coordinates constraints are not symetrical, the grid can be a rectangle,
 * we have to search for N "pixels" for one coordinate and M "pixels" on the
 * others. So that, events extraction is expressed as a extraction of events
 * uniformly distributed on a NxM grid.
 *
 * When doing an extraction, only the events with the lowest index are kept,
 * that's why we talk about "first" events in the next comments.
 *
 * To make the first events extraction efficient, each unsplitted quadtrees
 * nodes store events in ascending order relatively to their indices.
 */
template<int MAX_ELEMENTS_PER_NODE = 512, int REALLOC_ELEMENT_COUNT = 1000, int PREALLOC_ELEMENT_COUNT = 0, size_t Bbits = NBITS_INDEX>
class PVQuadTree
{
	constexpr static uint32_t mask_int_ycoord = (((uint32_t)1)<<Bbits)-1;

public:
	typedef PVTLRBuffer<Bbits> pv_tlr_buffer_t;
	typedef std::function<void(const PVQuadTreeEntry &entry, pv_tlr_buffer_t &buffer)> insert_entry_f;
	typedef std::function<bool(const PVQuadTreeEntry &entry, PVCore::PVHSVColor* image)> insert_entry_y1_y2_f;

public:
	/**
	 * Create and initialize a quadtree.
	 *
	 * @param y1_min_value the inclusive minimal bound along the y1 coordinate
	 * @param y1_max_value the exclusive maximal bound along the y1 coordinate
	 * @param y2_min_value the inclusive minimal bound along the y2 coordinate
	 * @param y2_max_value the exclusive maximal bound along the y2 coordinate
	 * @param max_level the depth limit to stop splitting the quadtres
	 */
	PVQuadTree(uint32_t y1_min_value, uint32_t y1_max_value, uint32_t y2_min_value, uint32_t y2_max_value, int max_level)
	{
		uint32_t y1_mid = y1_min_value + ((y1_max_value - y1_min_value) >> 1);
		uint32_t y2_mid = y2_min_value + ((y2_max_value - y2_min_value) >> 1);

		init(y1_min_value, y1_mid, y2_min_value, y2_mid, max_level);
	}

	/**
	 * Create a quadtree without initializing it, this constructor has to be used with init.
	 */
	PVQuadTree() : _nodes(nullptr), _index_min_bg(PVROW_INVALID_VALUE)
	{}

	/**
	 * Destruct a quadtree and its sub-quadtrees.
	 */
	~PVQuadTree() {
		if (_nodes == 0) {
			_datas.clear();
		} else {
			delete [] _nodes;
		}
	}

	/**
	 * Initialize a quadtree.
	 *
	 * @param y1_min_value the inclusive minimal bound along the y1 coordinate
	 * @param y1_mid_value the range middle value along the y1 coordinate
	 * @param y2_min_value the inclusive minimal bound along the y2 coordinate
	 * @param y2_mid_value the range middle value bound along the y2 coordinate
	 * @param max_level the depth limit to stop splitting the quadtres
	 */
	void init(uint32_t y1_min_value, uint32_t y1_mid_value, uint32_t y2_min_value, uint32_t y2_mid_value, int max_level)
	{
		_y1_min_value = y1_min_value;
		_y1_mid_value = y1_mid_value;
		_y2_min_value = y2_min_value;
		_y2_mid_value = y2_mid_value;
		_max_level = max_level;
		_index_min_bg = PVROW_INVALID_VALUE;
		if (PREALLOC_ELEMENT_COUNT != 0) {
			_datas.reserve(PREALLOC_ELEMENT_COUNT);
		} else {
			_datas = pvquadtree_entries_t();
		}
		_nodes = 0;
	}

	/**
	 * Insert an event.
	 *
	 * This method use the scholar algorithm because the non-linear memory accesses
	 * do not suit parallel algorithms.
	 *
	 * @param e the event to insert represented as a quadtree internal structure
	 */
	void insert(const PVQuadTreeEntry &e) {
		// searching for the right child
		register PVQuadTree *qt = this;
		while (qt->_nodes != 0) {
			qt = &qt->_nodes[qt->compute_index(e)];
		}

		// insertion
		qt->_datas.push_back(e);

		// does the current node must be splitted?
		if ((qt->_datas.size() >= MAX_ELEMENTS_PER_NODE)/* && (qt->_max_level > 0)*/) {
			qt->create_next_level();
		}
	}

	/**
	 * Reduce the memory usage.
	 *
	 * The internal structure (actually a PVVector) grows automatically and often store less
	 * entries that it can really contain.
	 *
	 * Due to memory managment performance, it is better to compact the quadtrees while a
	 * preprocessing pass.
	 *
	 * This method must not be inlined; some tests show that doing it implies important
	 * performance lose.
	 */
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

	/**
	 * Compute the memory used.
	 *
	 * @return the used memory
	 */
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

	/**
	 * Extract the first events according to constraints on y1 coordinate (y1_min, y1_max, zoom)
	 * and on y2 coordinate (y2_count).
	 *
	 * This method is initially (and principally) used to extract events for the right zone of a zoomed
	 * parallel view.
	 *
	 * @param y1_min the incluse minimal bound along the y1 coordinate
	 * @param y1_max the excluse maximal bound along the y1 coordinate
	 * @param zoom the zoom level
	 * @param y2_count the number of required events along the y2 coordinate
	 * @param buffer a temporary buffer used for event extraction
	 * @param insert_f the function executed for each relevant found event
	 * @param tls a TLR buffer used to store the result of extraction
	 */
	inline size_t get_first_from_y1(uint64_t y1_min, uint64_t y1_max, uint32_t zoom,
	                              uint32_t y2_count,
	                              pv_quadtree_buffer_entry_t *buffer,
	                              const insert_entry_f &insert_f,
	                              pv_tlr_buffer_t &tlr) const
	{
		return visit_y1::get_n_m(*this, y1_min, y1_max, zoom, y2_count,
		                  [](const PVQuadTreeEntry &e, const uint64_t y1_min, const uint64_t y1_max) -> bool
		                  {
			                  return (e.y1 >= y1_min) && (e.y1 < y1_max);
		                  },
		                  insert_f, buffer, tlr);
	}

	/**
	 * Extract the first events according to constraints on y2 coordinate (y2_min, y2_max, zoom)
	 * and on y1 coordinate (y1_count).
	 *
	 * This method is initially (and principally) used to extract events for the left zone of a zoomed
	 * parallel view.
	 *
	 * @param y2_min the incluse minimal bound along the y2 coordinate
	 * @param y2_max the excluse maximal bound along the y2 coordinate
	 * @param zoom the zoom level
	 * @param y1_count the number of required events along the y1 coordinate
	 * @param buffer a temporary buffer used for event extraction
	 * @param insert_f the function executed for each relevant found event
	 * @param tls a TLR buffer used to store the result of extraction
	 */
	inline size_t get_first_from_y2(uint64_t y2_min, uint64_t y2_max, uint32_t zoom,
	                              uint32_t y1_count,
	                              pv_quadtree_buffer_entry_t *buffer,
	                              const insert_entry_f &insert_f,
	                              pv_tlr_buffer_t &tlr) const
	{
		return visit_y2::get_n_m(*this, y2_min, y2_max, zoom, y1_count,
		                  [](const PVQuadTreeEntry &e, const uint64_t y2_min, const uint64_t y2_max) -> bool
		                  {
			                  return (e.y2 >= y2_min) && (e.y2 < y2_max);
		                  },
		                  insert_f, buffer, tlr);
	}

	/**
	 * Extract the first events according to constraints on both y1 and y2 coordinates (y1_min, y1_max, y2_min, y2_max, zoom)
	 *
	 * This method is initially (and principally) used to extract events for the right zone of a zoomed
	 * parallel view.
	 *
	 * @param y1_min the included minimal bound along the y1 coordinate
	 * @param y1_max the excluded maximal bound along the y1 coordinate
	 * @param y2_min the included minimal bound along the y2 coordinate
	 * @param y2_max the excluded maximal bound along the y2 coordinate
	 * @param zoom the zoom level
	 * @param buffer a temporary buffer used for event extraction
	 * @param insert_f the function executed for each relevant found event
	 * @param tls a TLR buffer used to store the result of extraction
	 */
	inline size_t get_first_from_y1_y2(
			uint64_t y1_min, uint64_t y1_max,
			uint64_t y2_min, uint64_t y2_max,
			uint32_t zoom,
			const double alpha,
			PVCore::PVHSVColor* image,
			const insert_entry_y1_y2_f &insert_f
			) const
		{
			return visit_y1_y2::get_n_m(*this, y1_min, y1_max, y2_min, y2_max, zoom, alpha,
			[](const PVQuadTreeEntry &e, const uint64_t y1_min, const uint64_t y1_max, const uint64_t y2_min, const uint64_t y2_max) -> bool
			{
				return (e.y1 >= y1_min) && (e.y1 < y1_max) && (e.y2 >= y2_min) && (e.y2 < y2_max);
			},
			insert_f, image);
		}

	inline size_t get_first_from_y1_y2_sel(
			uint64_t y1_min, uint64_t y1_max,
			uint64_t y2_min, uint64_t y2_max,
			uint32_t zoom,
			const double alpha,
			PVCore::PVHSVColor* image,
			const insert_entry_y1_y2_f &insert_f,
			Picviz::PVSelection const& sel
			) const
		{
			return visit_y1_y2::get_n_m_sel(*this, y1_min, y1_max, y2_min, y2_max, zoom, alpha,
			[](const PVQuadTreeEntry &e, const uint64_t y1_min, const uint64_t y1_max, const uint64_t y2_min, const uint64_t y2_max) -> bool
			{
				return (e.y1 >= y1_min) && (e.y1 < y1_max) && (e.y2 >= y2_min) && (e.y2 < y2_max);
			},
			insert_f, image, sel);
		}

	/**
	 * Extract the first selected events according to constraints on y1 coordinate (y1_min, y1_max,
	 * zoom) and on y2 coordinate (y2_count).
	 *
	 * This method is initially (and principally) used to extract events for the right zone of a zoomed
	 * parallel view.
	 *
	 * @param y1_min the incluse minimal bound along the y1 coordinate
	 * @param y1_max the excluse maximal bound along the y1 coordinate
	 * @param zoom the zoom level
	 * @param y2_count the number of required events along the y2 coordinate
	 * @param buffer a temporary buffer used for event extraction
	 * @param insert_f the function executed for each relevant found event
	 * @param tls a TLR buffer used to store the result of extraction
	 */
	inline size_t get_first_sel_from_y1(uint64_t y1_min, uint64_t y1_max,
	                                  const Picviz::PVSelection &selection,
	                                  uint32_t zoom, uint32_t y2_count,
	                                  pv_quadtree_buffer_entry_t *buffer,
	                                  const insert_entry_f &insert_f,
	                                  pv_tlr_buffer_t &tlr) const
	{
		return visit_y1::get_n_m(*this, y1_min, y1_max, zoom, y2_count,
		                  [&selection](const PVQuadTreeEntry &e, const uint64_t y1_min, const uint64_t y1_max) -> bool
		                  {
			                  return (e.y1 >= y1_min) && (e.y1 < y1_max)
				                  && selection.get_line(e.idx);
		                  },
		                  insert_f, buffer, tlr);
	}

	/**
	 * Extract the first selected events according to constraints on y2 coordinate (y2_min, y2_max,
	 * zoom) and on y1 coordinate (y1_count).
	 *
	 * This method is initially (and principally) used to extract events for the left zone of a zoomed
	 * parallel view.
	 *
	 * @param y2_min the incluse minimal bound along the y2 coordinate
	 * @param y2_max the excluse maximal bound along the y2 coordinate
	 * @param zoom the zoom level
	 * @param y1_count the number of required events along the y1 coordinate
	 * @param buffer a temporary buffer used for event extraction
	 * @param insert_f the function executed for each relevant found event
	 * @param tls a TLR buffer used to store the result of extraction
	 */
	inline size_t get_first_sel_from_y2(uint64_t y2_min, uint64_t y2_max,
	                                  const Picviz::PVSelection &selection,
	                                  uint32_t zoom, uint32_t y1_count,
	                                  pv_quadtree_buffer_entry_t *buffer,
	                                  const insert_entry_f &insert_f,
	                                  pv_tlr_buffer_t &tlr) const
	{
		return visit_y2::get_n_m(*this, y2_min, y2_max, zoom, y1_count,
		                  [&selection](const PVQuadTreeEntry &e, const uint64_t y2_min, const uint64_t y2_max) -> bool
		                  {
			                  return (e.y2 >= y2_min) && (e.y2 < y2_max)
				                  && selection.get_line(e.idx);
		                  },
		                  insert_f, buffer, tlr);
	}

	/**
	 * Search for all events whose y1 coordinates are in in the range [y1_min,y1_max) and
	 * mark them as selected in \selection.
	 *
	 * There is no constraint on the y1 coordinate.
	 *
	 * @param y1_min the incluse minimal bound along the y1 coordinate
	 * @param y1_max the excluse maximal bound along the y1 coordinate
	 * @param selection the structure containing the result
	 *
	 * @return the number of selected events
	 */
	size_t compute_selection_y1(const uint64_t y1_min, const uint64_t y1_max, Picviz::PVSelection &selection) const
	{
		return compute_selection_y1(*this, y1_min, y1_max, selection);
	}

	/**
	 * Search for all events whose y2 coordinates are in the range [y2_min,y2_max) and
	 * mark them as selected in \selection.
	 *
	 * There is no constraint on the y1 coordinate.
	 *
	 * @param y2_min the incluse minimal bound along the y2 coordinate
	 * @param y2_max the excluse maximal bound along the y2 coordinate
	 * @param selection the structure containing the result
	 *
	 * @return the number of selected events
	 */
	size_t compute_selection_y2(const uint64_t y2_min, const uint64_t y2_max, Picviz::PVSelection &selection) const
	{
		return compute_selection_y2(*this, y2_min, y2_max, selection);
	}

	void set_min_idx_sel_invalid()
	{
		_index_min_sel = PVROW_INVALID_VALUE;
		if (_nodes) {
			for (int i = 0; i < 4; i++) {
				_nodes[i].set_min_idx_sel_invalid();
			}
		}
	}

	PVRow compute_min_indexes_sel_notempty(Picviz::PVSelection const& sel)
	{
		PVRow idx_min;

		if (!_nodes) {
			idx_min = PVROW_INVALID_VALUE;
			for (size_t i = 0; i < _datas.size(); i++) {
				const PVRow idx = _datas.at(i).idx;
				if (sel.get_line_fast(idx)) {
					idx_min = idx;
					break;
				}
			}
		}
		else {
			__m128i mins = _mm_set_epi32(_nodes[0].compute_min_indexes_sel_notempty(sel),
			                             _nodes[1].compute_min_indexes_sel_notempty(sel),
			                             _nodes[2].compute_min_indexes_sel_notempty(sel),
			                             _nodes[3].compute_min_indexes_sel_notempty(sel));
			idx_min = picviz_mm_hmin_epu32(mins);
		}

		_index_min_sel = idx_min;
		return idx_min;
	}

	/*
	PVRow compute_min_indexes_bg(Picviz::PVSelection const& layers_sel)
	{
		PVRow idx_min;
		if (!_nodes) {
			idx_min = _datas.at(0);
			if (!layers_sel.is_empty()) {
				for (size_t i = 0; i < _datas.size(); i++) {
					const PVRow idx = _datas.at(i).idx;
					if (layers_sel.get_line_fast(idx)) {
						idx_min = idx;
						break;
					}
				}
			}
		}
		else {
			__m128i mins = _mm_insert_epi32(_mm_setzero_si128(), _nodes[0].compute_min_indexes_bg(layers_sel), 0);
			mins = _mm_insert_epi32(mins, _nodes[1].compute_min_indexes_bg(layers_sel), 1);
			mins = _mm_insert_epi32(mins, _nodes[2].compute_min_indexes_bg(layers_sel), 2);
			mins = _mm_insert_epi32(mins, _nodes[3].compute_min_indexes_bg(layers_sel), 3);
			idx_min = picviz_mm_hmin_epu32(mins);
		}

		_index_min_sel = idx_min;
		return idx_min;
	}*/

	/**
	 * Equality test.
	 *
	 * @param qt the second quadtree
	 *
	 * @return true if the 2 quadtrees have the same structure and the
	 * same content; false otherwise.
	 */
	bool operator==(const PVQuadTree &qt) const
	{
		if ((_y1_min_value != qt._y1_min_value) || (_y1_mid_value != qt._y1_mid_value) || (_y2_min_value != qt._y2_min_value) || (_y2_mid_value != qt._y2_mid_value)) {
			return false;
		} else if (((_nodes != nullptr) != (qt._nodes != nullptr))) {
			return false;
		} else if ((_nodes != nullptr) && (qt._nodes != nullptr)) {
			// compare the sub-quadtrees
			for(int i = 0; i < 4; ++i) {
				if (_nodes[i] == qt._nodes[i]) {
					continue;
				}
				return false;
			}
			return true;
		} else if (_datas == qt._datas) {
			// compare the vectors
			return true;
		} else {
			return false;
		}
	}

	/**
	 * Save the quadtree into a file.
	 *
	 * @param filename the output filename
	 *
	 * @return true on success; false otherwise and an error is printed.
	 */
	bool dump_to_file(const char *filename) const
	{
		FILE *fp = fopen(filename, "w");
		if (fp == NULL) {
			PVLOG_ERROR("Error while opening %s for writing: %s.\n",
			            filename, strerror(errno));
			return false;
		}

		bool ret = dump_to_file(filename, fp);

		fclose(fp);
		return ret;
	}

	/**
	 * Dump the quadtree to a FILE*.
	 *
	 * @param filename the output filename
	 * @param fp the opened output FILE*
	 *
	 * @return true on success; false otherwise and an error is printed.
	 */
	bool dump_to_file(const char *filename, FILE *fp) const
	{
		if (fwrite(&_y1_min_value, sizeof(_y1_min_value), 1, fp) != 1) {
			PVLOG_ERROR("Error while writing %s: %s.\n",
			            filename, strerror(errno));
			return false;
		}

		if (fwrite(&_y1_mid_value, sizeof(_y1_mid_value), 1, fp) != 1) {
			PVLOG_ERROR("Error while writing %s: %s.\n",
			            filename, strerror(errno));
			return false;
		}

		if (fwrite(&_y2_min_value, sizeof(_y2_min_value), 1, fp) != 1) {
			PVLOG_ERROR("Error while writing %s: %s.\n",
			            filename, strerror(errno));
			return false;
		}

		if (fwrite(&_y2_mid_value, sizeof(_y2_mid_value), 1, fp) != 1) {
			PVLOG_ERROR("Error while writing %s: %s.\n",
			            filename, strerror(errno));
			return false;
		}

		if (fwrite(&_max_level, sizeof(_max_level), 1, fp) != 1) {
			PVLOG_ERROR("Error while writing %s: %s.\n",
			            filename, strerror(errno));
			return false;
		}

		int splitted = (_nodes != nullptr);
		if (fwrite(&splitted, sizeof(splitted), 1, fp) != 1) {
			PVLOG_ERROR("Error while writing %s: %s.\n",
			            filename, strerror(errno));
			return false;
		}

		if (splitted) {
			for(int i = 0; i < 4; ++i) {
				if (_nodes[i].dump_to_file(filename, fp) == false) {
					return false;
				}
			}
		} else {
			pvquadtree_entries_t::size_type count = _datas.size();

			if (fwrite(&count, sizeof(count), 1, fp) != 1) {
				PVLOG_ERROR("Error while writing %s: %s.\n",
				            filename, strerror(errno));
				return false;
			}

			if (count != 0) {
				size_t size_elt = sizeof(pvquadtree_entries_t::value_type);

				if (fwrite(_datas.get_pointer(), size_elt, count, fp) != count) {
					PVLOG_ERROR("Error while writing %s: %s.\n",
					            filename, strerror(errno));
					return false;
				}
			}
		}

		return true;
	}

	/**
	 * Load the quadtree from a file.
	 *
	 * @param filename the input filename
	 *
	 * @return true on success; false otherwise and an error is printed.
	 */
	static PVQuadTree *load_from_file(const char *filename)
	{
		FILE *fp = fopen(filename, "r");
		if (fp == nullptr) {
			PVLOG_ERROR("Error while opening %s for reading: %s.\n",
			            filename, strerror(errno));
			return nullptr;
		}

		PVQuadTree* qt = new PVQuadTree();
		bool ret = qt->load_from_file(filename, fp);

		fclose(fp);

		if (ret == false) {
			delete qt;
			qt = nullptr;
		}
		return qt;
	}

	/**
	 * Load recursively quadtree from a FILE*.
	 *
	 * @param filename the input filename
	 * @param fp the opened input FILE*
	 *
	 * @return true on success; false otherwise and an error is printed.
	 */
	bool load_from_file(const char *filename, FILE *fp)
	{
		if (fread(&_y1_min_value, sizeof(_y1_min_value), 1, fp) != 1) {
			PVLOG_ERROR("Error while reading %s: %s.\n",
			            filename, strerror(errno));
			return false;
		}

		if (fread(&_y1_mid_value, sizeof(_y1_mid_value), 1, fp) != 1) {
			PVLOG_ERROR("Error while reading %s: %s.\n",
			            filename, strerror(errno));
			return false;
		}

		if (fread(&_y2_min_value, sizeof(_y2_min_value), 1, fp) != 1) {
			PVLOG_ERROR("Error while reading %s: %s.\n",
			            filename, strerror(errno));
			return false;
		}

		if (fread(&_y2_mid_value, sizeof(_y2_mid_value), 1, fp) != 1) {
			PVLOG_ERROR("Error while reading %s: %s.\n",
			            filename, strerror(errno));
			return false;
		}

		if (fread(&_max_level, sizeof(_max_level), 1, fp) != 1) {
			PVLOG_ERROR("Error while reading %s: %s.\n",
			            filename, strerror(errno));
			return false;
		}

		int splitted;
		if (fread(&splitted, sizeof(splitted), 1, fp) != 1) {
			PVLOG_ERROR("Error while reading %s: %s.\n",
			            filename, strerror(errno));
			return false;
		}

		if (splitted) {
			_nodes = new PVQuadTree [4];
			for(int i = 0; i < 4; ++i) {
				if (_nodes[i].load_from_file(filename, fp) == false) {
					return false;
				}
			}
		} else {
			pvquadtree_entries_t::size_type count;

			if (fread(&count, sizeof(count), 1, fp) != 1) {
				PVLOG_ERROR("Error while reading %s: %s.\n",
				            filename, strerror(errno));
				return false;
			}

			if (count != 0) {
				_datas.reserve(count);

				size_t size_elt = sizeof(pvquadtree_entries_t::value_type);
				if (fread(_datas.get_pointer(), size_elt, count, fp) != count) {
					PVLOG_ERROR("Error while reading %s: %s.\n",
					            filename, strerror(errno));
					return false;
				}

				_datas.set_index(count);
			} else {
				_datas.clear();
			}
		}
		return true;
	}

#ifdef PICVIZ_DEVELOPER_MODE
	
	/**
	 * Clear the chronometer used to measure the whole extraction process time.
	 */
	static void all_clear()
	{
		extract_stat::all_dt = 0;
	}

	/**
	 * Read the chronometer used to measure the whole extraction process time.
	 *
	 * For test purpose only.
	 */
	static double all_get()
	{
		return extract_stat::all_dt;
	}

	/**
	 * Clear the parsed events counter used while extraction.
	 *
	 * For test purpose only.
	 */
	static void all_count_clear()
	{
		extract_stat::all_cnt = 0;
	}

	/**
	 * Read the parsed events counter used while extraction.
	 *
	 * For test purpose only.
	 */
	static size_t all_count_get()
	{
		return extract_stat::all_cnt;
	}

	/**
	 * Clear the tested events counter used while extraction.
	 *
	 * For test purpose only.
	 */
	static void test_count_clear()
	{
		extract_stat::test_cnt = 0;
	}

	/**
	 * Read the tested events counter used while extraction.
	 *
	 * For test purpose only.
	 */
	static size_t test_count_get()
	{
		return extract_stat::test_cnt;
	}

	/**
	 * Clear the inserted events counter used while extraction.
	 *
	 * For test purpose only.
	 */
	static void insert_count_clear()
	{
		extract_stat::insert_cnt = 0;
	}

	/**
	 * Read the inserted events counter used while extraction.
	 *
	 * For test purpose only.
	 */
	static size_t insert_count_get()
	{
		return extract_stat::insert_cnt;
	}
#endif

private:
	/**
	 * Initialize a quadtree given an other quadtree.
	 *
	 * @param qt the quadtree from which parameters are copied
	 */
	void init(const PVQuadTree &qt)
	{
		_y1_min_value = qt._y1_min_value;
		_y1_mid_value = qt._y1_mid_value;
		_y2_min_value = qt._y2_min_value;
		_y2_mid_value = qt._y2_mid_value;
		_max_level = qt._max_level;
		_nodes = 0;
	}

	/**
	 * Compute the sub-quadtree's index corresponding to an entry.
	 *
	 * @param e the event to test
	 *
	 * @return the subtree's index corresponding to \e
	 */
	inline int compute_index(const PVQuadTreeEntry &e) const
	{
		return ((e.y2 > _y2_mid_value) << 1) | (e.y1 > _y1_mid_value);
	}

	/**
	 * Subdivide a non-splitted quadtree into 4 non-splitted sub-quadtrees.
	 */
	void create_next_level()
	{
		assert(_datas.size() > 0);

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

	uint32_t get_first_elt_index() const
	{
		if (_index_min_bg != PVROW_INVALID_VALUE) {
			return _index_min_bg;
		}
		if (_datas.size() > 0) {
			return _datas.at(0).idx;
		}
		return PVROW_INVALID_VALUE;
	}


private:
	/**
	 * @struct visit_y1
	 *
	 * Visitor used for quadtree traversal using strong constraints on y1 coordinate.
	 */
	struct visit_y1
	{
		/**
		 * Traversal function to extract a NxM events grid.
		 *
		 * @param obj the quadtree to visit
		 * @param y1_min the incluse minimal bound along the y1 coordinate
		 * @param y1_max the excluse maximal bound along the y1 coordinate
		 * @param zoom the zoom level
		 * @param y2_count the number of required events along the y2 coordinate
		 * @param test_f the function executed to test if an event is relevant or not
		 * @param insert_f the function executed for each relevant found event
		 * @param buffer a temporary buffer used for event extraction
		 * @param tls a TLR buffer used to store the result of extraction
		 */
		template <typename Ftest>
		static size_t get_n_m(PVQuadTree const& obj,
		                    const uint64_t y1_min, const uint64_t y1_max,
		                    const uint32_t zoom, const uint32_t y2_count,
		                    const Ftest &test_f, const insert_entry_f &insert_f,
		                    pv_quadtree_buffer_entry_t *buffer,
		                    pv_tlr_buffer_t &tlr)
		{
			size_t ret = 0;
			if (zoom == 0) {
				/* need a 1 entry width band along y1
				 */
				ret += get_1_m(obj, y1_min, y1_max, y2_count, test_f, insert_f, buffer, tlr);
			} else if (y2_count == 1) {
				/* need a 1 entry width band along y2
				 */
				ret += get_n_1(obj, y1_min, y1_max, zoom, test_f, insert_f, buffer, tlr);
			} else if (obj._nodes != 0) {
				/* recursive search can be processed
				 */
				if (obj._y1_mid_value < y1_max) {
					ret += get_n_m(obj._nodes[NE], y1_min, y1_max,
					        zoom - 1, y2_count >> 1,
					        test_f, insert_f, buffer, tlr);
					ret += get_n_m(obj._nodes[SE], y1_min, y1_max,
					        zoom - 1, y2_count >> 1,
					        test_f, insert_f, buffer, tlr);
				}
				if (y1_min < obj._y1_mid_value) {
					ret += get_n_m(obj._nodes[NW], y1_min, y1_max,
					        zoom - 1, y2_count >> 1,
					        test_f, insert_f, buffer, tlr);
					ret += get_n_m(obj._nodes[SW], y1_min, y1_max,
					        zoom - 1, y2_count >> 1,
					        test_f, insert_f, buffer, tlr);
				}
			} else if (obj._datas.size() != 0) {
				/* this is a unsplitted node with data and an array of nxm
				 * entries is needed
				 */
#ifdef QUADTREE_USE_SSE_EXTRACT
				ret += extract_sse(obj, y1_min, y1_max, zoom, y2_count,
				            test_f, insert_f, buffer, tlr);
#else
				ret += extract_seq(obj, y1_min, y1_max, zoom, y2_count,
				            test_f, insert_f, buffer, tlr);
#endif
			}
			return ret;
		}

		/**
		 * Traversal function to extract a 1xM events grid.
		 *
		 * @param obj the quadtree to visit
		 * @param y1_min the incluse minimal bound along the y1 coordinate
		 * @param y1_max the excluse maximal bound along the y1 coordinate
		 * @param y2_count the number of required events along the y2 coordinate
		 * @param test_f the function executed to test if an event is relevant or not
		 * @param insert_f the function executed for each relevant found event
		 * @param buffer a temporary buffer used for event extraction
		 * @param tls a TLR buffer used to store the result of extraction
		 */
		template <typename Ftest>
		static size_t get_1_m(PVQuadTree const& obj,
		                    const uint64_t y1_min, const uint64_t y1_max,
		                    const uint32_t y2_count,
		                    const Ftest &test_f,
		                    const insert_entry_f &insert_f,
		                    pv_quadtree_buffer_entry_t *buffer,
		                    pv_tlr_buffer_t &tlr)
		{
			size_t ret = 0;
			if (y2_count == 1) {
				/* time to extract
				 */
				PVQuadTreeEntry e(0, 0, UINT32_MAX);
				get_1_1(obj, y1_min, y1_max, test_f, e);
				if (e.idx != UINT32_MAX) {
					insert_f(e, tlr);
				}
				ret = 1;
			} else  if (obj._nodes != 0) {
				if (obj._y1_mid_value < y1_max) {
					ret += get_1_m(obj._nodes[NE], y1_min, y1_max,
					        y2_count >> 1, test_f, insert_f,
					        buffer, tlr);
					ret += get_1_m(obj._nodes[SE], y1_min, y1_max,
					        y2_count >> 1, test_f, insert_f,
					        buffer, tlr);
				}
				if (y1_min < obj._y1_mid_value) {
					ret += get_1_m(obj._nodes[NW], y1_min, y1_max,
					        y2_count >> 1, test_f, insert_f,
					        buffer, tlr);
					ret += get_1_m(obj._nodes[SW], y1_min, y1_max,
					        y2_count >> 1, test_f, insert_f,
					        buffer, tlr);
				}
			} else if (obj._datas.size() != 0) {
				/* this is a unsplitted node with data and an array of 1xm
				 * entries is needed
				 */
#ifdef QUADTREE_USE_SSE_EXTRACT
				ret += extract_sse(obj, y1_min, y1_max, 0, y2_count,
				            test_f, insert_f, buffer, tlr);
#else
				ret += extract_seq(obj, y1_min, y1_max, 0, y2_count,
				            test_f, insert_f, buffer, tlr);
#endif
			}
			return ret;
		}

		/**
		 * Traversal function to extract a Nx1 events grid.
		 *
		 * @param obj the quadtree to visit
		 * @param y1_min the incluse minimal bound along the y1 coordinate
		 * @param y1_max the excluse maximal bound along the y1 coordinate
		 * @param zoom the zoom level
		 * @param test_f the function executed to test if an event is relevant or not
		 * @param insert_f the function executed for each relevant found event
		 * @param buffer a temporary buffer used for event extraction
		 * @param tls a TLR buffer used to store the result of extraction
		 */
		template <typename Ftest>
		static size_t get_n_1(PVQuadTree const& obj,
		                    const uint64_t y1_min, const uint64_t y1_max,
		                    const uint32_t zoom,
		                    const Ftest &test_f,
		                    const insert_entry_f &insert_f,
		                    pv_quadtree_buffer_entry_t *buffer,
		                    pv_tlr_buffer_t &tlr)
		{
			size_t ret = 0;
			if (zoom == 0) {
				/* time to extract
				 */
				PVQuadTreeEntry e(0, 0, UINT32_MAX);
				get_1_1(obj, y1_min, y1_max, test_f, e);
				if (e.idx != UINT32_MAX) {
					insert_f(e, tlr);
				}
				ret = 1;
			} else if (obj._nodes != 0) {
				if (obj._y1_mid_value < y1_max) {
					ret += get_n_1(obj._nodes[NE], y1_min, y1_max,
					        zoom - 1, test_f, insert_f,
					        buffer, tlr);
					ret += get_n_1(obj._nodes[SE], y1_min, y1_max,
					        zoom - 1, test_f, insert_f,
					        buffer, tlr);
				}
				if (y1_min < obj._y1_mid_value) {
					ret += get_n_1(obj._nodes[NW], y1_min, y1_max,
					        zoom - 1, test_f, insert_f,
					        buffer, tlr);
					ret += get_n_1(obj._nodes[SW], y1_min, y1_max,
					        zoom - 1, test_f, insert_f,
					        buffer, tlr);
				}
			} else if (obj._datas.size() != 0) {
				/* this is a unsplitted node with data and an array of nx1
				 * entries is needed
				 */
#ifdef QUADTREE_USE_SSE_EXTRACT
				ret += extract_sse(obj, y1_min, y1_max, zoom, 1,
				            test_f, insert_f, buffer, tlr);
#else
				ret += extract_seq(obj, y1_min, y1_max, zoom, 1,
				            test_f, insert_f, buffer, tlr);
#endif
			}
			return ret;
		}

		/**
		 * Traversal function to extract only one event for a quadtree.
		 *
		 * @param obj the quadtree to visit
		 * @param y1_min the incluse minimal bound along the y1 coordinate
		 * @param y1_max the excluse maximal bound along the y1 coordinate
		 * @param test_f the function executed to test if an event is relevant or not
		 * @param insert_f the function executed for each relevant found event
		 */
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

		/**
		 * Sequential function to extract a NxM events grid from a non-splitted quadtree.
		 *
		 * @param obj the quadtree to visit
		 * @param y1_min the incluse minimal bound along the y1 coordinate
		 * @param y1_max the excluse maximal bound along the y1 coordinate
		 * @param zoom the zoom level
		 * @param y2_count the number of required events along the y2 coordinate
		 * @param test_f the function executed to test if an event is relevant or not
		 * @param insert_f the function executed for each relevant found event
		 * @param buffer a temporary buffer used for event extraction
		 * @param tls a TLR buffer used to store the result of extraction
		 */
		template <typename Ftest>
		static size_t extract_seq(PVQuadTree const& obj,
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
			const uint64_t y1_scale = PVCore::max(1UL, y1_len / max_count);

			const uint64_t y2_orig = obj._y2_min_value;
			const uint64_t y2_scale = ((obj._y2_mid_value - y2_orig) * 2) / y2_count;

			const uint64_t ly1_min = (PVCore::clamp(y1_min, y1_orig, y1_orig + y1_len) - y1_orig) / y1_scale;
			const uint64_t ly1_max = (PVCore::clamp(y1_max, y1_orig, y1_orig + y1_len) - y1_orig) / y1_scale;
			const uint64_t clipped_max_count = PVCore::max(1UL, ly1_max - ly1_min);
			const size_t count_aligned = ((clipped_max_count * y2_count) + 31) / 32;
			memset(buffer, 0, count_aligned * sizeof(uint32_t));
			uint32_t remaining = clipped_max_count * y2_count;

			size_t ninsert = 0;
			for(size_t i = 0; i < obj._datas.size(); ++i) {
#ifdef PICVIZ_DEVELOPER_MODE
				++extract_stat::all_cnt;
#endif
				const PVQuadTreeEntry &e = obj._datas.at(i);
				if (!test_f(e, y1_min, y1_max)) {
					continue;
				}
				const uint32_t pos = (((e.y1 - y1_orig) / y1_scale) - ly1_min) + clipped_max_count * ((e.y2 - y2_orig) / y2_scale);
#ifdef PICVIZ_DEVELOPER_MODE
				++extract_stat::test_cnt;
#endif
				if (B_IS_SET(buffer[pos >> 5], pos & 31)) {
					continue;
				}
				insert_f(e, tlr);
#ifdef PICVIZ_DEVELOPER_MODE
				++extract_stat::insert_cnt;
#endif
				ninsert++;
				B_SET(buffer[pos >> 5], pos & 31);
				--remaining;
				if (remaining == 0) {
					break;
				}
			}
			BENCH_STOP(extract);
#ifdef PICVIZ_DEVELOPER_MODE
			extract_stat::all_dt += BENCH_END_TIME(extract);
#endif
			return ninsert;
		}

		/**
		 * SSE function to extract a NxM events grid from a non-splitted quadtree.
		 *
		 * @param obj the quadtree to visit
		 * @param y1_min the incluse minimal bound along the y1 coordinate
		 * @param y1_max the excluse maximal bound along the y1 coordinate
		 * @param zoom the zoom level
		 * @param y2_count the number of required events along the y2 coordinate
		 * @param test_f the function executed to test if an event is relevant or not
		 * @param insert_f the function executed for each relevant found event
		 * @param buffer a temporary buffer used for event extraction
		 * @param tls a TLR buffer used to store the result of extraction
		 */
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
			const uint64_t y1_scale = PVCore::max(1UL, y1_len / max_count);
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

	/**
	 * @struct visit_y2
	 *
	 * Visitor used for quadtree traversal using strong constraints on y2 coordinate.
	 */
	struct visit_y2
	{
		/**
		 * Traversal function to extract a NxM events grid.
		 *
		 * @param obj the quadtree to visit
		 * @param y2_min the incluse minimal bound along the y2 coordinate
		 * @param y2_max the excluse maximal bound along the y2 coordinate
		 * @param zoom the zoom level
		 * @param y1_count the number of required events along the y1 coordinate
		 * @param test_f the function executed to test if an event is relevant or not
		 * @param insert_f the function executed for each relevant found event
		 * @param buffer a temporary buffer used for event extraction
		 * @param tls a TLR buffer used to store the result of extraction
		 */
		template <typename Ftest>
		static size_t get_n_m(PVQuadTree const& obj,
		                    const uint64_t y2_min, const uint64_t y2_max,
		                    const uint32_t zoom, const uint32_t y1_count,
		                    const Ftest &test_f,
		                    const insert_entry_f &insert_f,
		                    pv_quadtree_buffer_entry_t *buffer,
		                    pv_tlr_buffer_t &tlr)
		{
			size_t ret = 0;
			if (zoom == 0) {
				/* need a 1 entry width band along y2
				 */
				ret += get_1_m(obj, y2_min, y2_max, y1_count, test_f, insert_f, buffer, tlr);
			} else if (y1_count == 1) {
				/* need a 1 entry width band along y1
				 */
				ret += get_n_1(obj, y2_min, y2_max, zoom, test_f, insert_f, buffer, tlr);
			} else if (obj._nodes != 0) {
				/* recursive search can be processed
				 */
				if (obj._y2_mid_value < y2_max) {
					ret += get_n_m(obj._nodes[NE], y2_min, y2_max,
					        zoom - 1, y1_count >> 1,
					        test_f, insert_f, buffer, tlr);
					ret += get_n_m(obj._nodes[NW], y2_min, y2_max,
					        zoom - 1, y1_count >> 1,
					        test_f, insert_f, buffer, tlr);
				}
				if (y2_min < obj._y2_mid_value) {
					ret += get_n_m(obj._nodes[SE], y2_min, y2_max,
					        zoom - 1, y1_count >> 1,
					        test_f, insert_f, buffer, tlr);
					ret += get_n_m(obj._nodes[SW], y2_min, y2_max,
					        zoom - 1, y1_count >> 1,
					        test_f, insert_f, buffer, tlr);
				}
			} else if (obj._datas.size() != 0) {
				/* this is a unsplitted node with data and an array of nxm
				 * entries is needed
				 */
#ifdef QUADTREE_USE_SSE_EXTRACT
				ret += extract_sse(obj, y2_min, y2_max, zoom, y1_count,
				            test_f, insert_f, buffer, tlr);
#else
				ret += extract_seq(obj, y2_min, y2_max, zoom, y1_count,
				            test_f, insert_f, buffer, tlr);
#endif
			}
			return ret;
		}

		/**
		 * Traversal function to extract a 1xM events grid.
		 *
		 * @param obj the quadtree to visit
		 * @param y2_min the incluse minimal bound along the y2 coordinate
		 * @param y2_max the excluse maximal bound along the y2 coordinate
		 * @param y1_count the number of required events along the y1 coordinate
		 * @param test_f the function executed to test if an event is relevant or not
		 * @param insert_f the function executed for each relevant found event
		 * @param buffer a temporary buffer used for event extraction
		 * @param tls a TLR buffer used to store the result of extraction
		 */
		template <typename Ftest>
		static size_t get_1_m(PVQuadTree const& obj,
		                    const uint64_t y2_min, const uint64_t y2_max,
		                    const uint32_t y1_count,
		                    const Ftest &test_f,
		                    const insert_entry_f &insert_f,
		                    pv_quadtree_buffer_entry_t *buffer,
		                    pv_tlr_buffer_t &tlr)
		{
			size_t ret = 0;
			if (y1_count == 1) {
				/* time to extract
				 */
				PVQuadTreeEntry e(0, 0, UINT32_MAX);
				get_1_1(obj, y2_min, y2_max, test_f, e);
				if (e.idx != UINT32_MAX) {
					insert_f(e, tlr);
				}
				ret = 1;
			} else  if (obj._nodes != 0) {
				if (obj._y2_mid_value < y2_max) {
					ret += get_1_m(obj._nodes[NE], y2_min, y2_max,
					        y1_count >> 1, test_f, insert_f,
					        buffer, tlr);
					ret += get_1_m(obj._nodes[NW], y2_min, y2_max,
					        y1_count >> 1, test_f, insert_f,
					        buffer, tlr);
				}
				if (y2_min < obj._y2_mid_value) {
					ret += get_1_m(obj._nodes[SE], y2_min, y2_max,
					        y1_count >> 1, test_f, insert_f,
					        buffer, tlr);
					ret += get_1_m(obj._nodes[SW], y2_min, y2_max,
					        y1_count >> 1, test_f, insert_f,
					        buffer, tlr);
				}
			} else if (obj._datas.size() != 0) {
				/* this is a unsplitted node with data and an array of 1xm
				 * entries is needed
				 */
#ifdef QUADTREE_USE_SSE_EXTRACT
				ret += extract_sse(obj, y2_min, y2_max, 0, y1_count,
				            test_f, insert_f, buffer, tlr);
#else
				ret += extract_seq(obj, y2_min, y2_max, 0, y1_count,
				            test_f, insert_f, buffer, tlr);
#endif
			}
			return ret;
		}

		/**
		 * Traversal function to extract a Nx1 events grid.
		 *
		 * @param obj the quadtree to visit
		 * @param y2_min the incluse minimal bound along the y2 coordinate
		 * @param y2_max the excluse maximal bound along the y2 coordinate
		 * @param zoom the zoom level
		 * @param test_f the function executed to test if an event is relevant or not
		 * @param insert_f the function executed for each relevant found event
		 * @param buffer a temporary buffer used for event extraction
		 * @param tls a TLR buffer used to store the result of extraction
		 */
		template <typename Ftest>
		static size_t get_n_1(PVQuadTree const& obj,
		                    const uint64_t y2_min, const uint64_t y2_max,
		                    const uint32_t zoom,
		                    const Ftest &test_f,
		                    const insert_entry_f &insert_f,
		                    pv_quadtree_buffer_entry_t *buffer,
		                    pv_tlr_buffer_t &tlr)
		{
			size_t ret = 0;
			if (zoom == 0) {
				/* time to extract
				 */
				PVQuadTreeEntry e(0, 0, UINT32_MAX);
				get_1_1(obj, y2_min, y2_max, test_f, e);
				if (e.idx != UINT32_MAX) {
					insert_f(e, tlr);
				}
				ret = 1;
			} else if (obj._nodes != 0) {
				if (obj._y2_mid_value < y2_max) {
					ret += get_n_1(obj._nodes[NE], y2_min, y2_max,
					        zoom - 1, test_f, insert_f, buffer, tlr);
					ret += get_n_1(obj._nodes[NW], y2_min, y2_max,
					        zoom - 1, test_f, insert_f, buffer, tlr);
				}
				if (y2_min < obj._y2_mid_value) {
					ret += get_n_1(obj._nodes[SE], y2_min, y2_max,
					        zoom - 1, test_f, insert_f, buffer, tlr);
					ret += get_n_1(obj._nodes[SW], y2_min, y2_max,
					        zoom - 1, test_f, insert_f, buffer, tlr);
				}
			} else if (obj._datas.size() != 0) {
				/* this is a unsplitted node with data and an array of nx1
				 * entries is needed
				 */
#ifdef QUADTREE_USE_SSE_EXTRACT
				ret += extract_sse(obj, y2_min, y2_max, zoom, 1,
				            test_f, insert_f, buffer, tlr);
#else
				ret += extract_seq(obj, y2_min, y2_max, zoom, 1,
				            test_f, insert_f, buffer, tlr);
#endif
			}
			return ret;
		}

		/**
		 * Traversal function to extract only one event for a quadtree.
		 *
		 * @param obj the quadtree to visit
		 * @param y2_min the incluse minimal bound along the y2 coordinate
		 * @param y2_max the excluse maximal bound along the y2 coordinate
		 * @param test_f the function executed to test if an event is relevant or not
		 */
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

		/**
		 * Sequential function to extract a NxM events grid from a non-splitted quadtree.
		 *
		 * @param obj the quadtree to visit
		 * @param y2_min the incluse minimal bound along the y2 coordinate
		 * @param y2_max the excluse maximal bound along the y2 coordinate
		 * @param zoom the zoom level
		 * @param y1_count the number of required events along the y1 coordinate
		 * @param test_f the function executed to test if an event is relevant or not
		 * @param insert_f the function executed for each relevant found event
		 * @param buffer a temporary buffer used for event extraction
		 * @param tls a TLR buffer used to store the result of extraction
		 */
		template <typename Ftest>
		static size_t extract_seq(PVQuadTree const& obj,
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
			const uint64_t y2_scale = PVCore::max(1UL, y2_len / max_count);
			const uint64_t ly2_min = (PVCore::clamp(y2_min, y2_orig, y2_orig + y2_len) - y2_orig) / y2_scale;
			const uint64_t ly2_max = (PVCore::clamp(y2_max, y2_orig, y2_orig + y2_len) - y2_orig) / y2_scale;
			const uint64_t clipped_max_count = PVCore::max(1UL, ly2_max - ly2_min);
			const size_t count_aligned = ((clipped_max_count * y1_count) + 31) / 32;
			memset(buffer, 0, count_aligned * sizeof(uint32_t));
			uint32_t remaining = clipped_max_count * y1_count;
			size_t ret = 0;
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
				ret++;
				B_SET(buffer[pos >> 5], pos & 31);
				--remaining;
				if (remaining == 0) {
					break;
				}
			}
			return ret;
		}

		/**
		 * SSE function to extract a NxM events grid from a non-splitted quadtree.
		 *
		 * @param obj the quadtree to visit
		 * @param y2_min the incluse minimal bound along the y2 coordinate
		 * @param y2_max the excluse maximal bound along the y2 coordinate
		 * @param zoom the zoom level
		 * @param y1_count the number of required events along the y1 coordinate
		 * @param test_f the function executed to test if an event is relevant or not
		 * @param insert_f the function executed for each relevant found event
		 * @param buffer a temporary buffer used for event extraction
		 * @param tls a TLR buffer used to store the result of extraction
		 */
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
			const uint64_t y2_scale = PVCore::max(1UL, y2_len / max_count);
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

	/**
	 * @struct visit_y1_y2
	 *
	 * Visitor used for quadtree traversal using strong constraints on both y1 and y2 coordinates.
	 */
	struct visit_y1_y2
	{
		/**
		 * Traversal function to extract a NxM events grid.
		 *
		 * @param obj the quadtree to visit
		 * @param y1_min the included minimal bound along the y1 coordinate
		 * @param y1_max the excluded maximal bound along the y1 coordinate
		 * @param y2_min the included minimal bound along the y2 coordinate
		 * @param y2_max the excluded maximal bound along the y2 coordinate
		 * @param zoom the zoom level
		 * @param test_f the function executed to test if an event is relevant or not
		 * @param insert_f the function executed for each relevant found event
		 * @param image the output PVHSVColor image
		 */
		template <typename Ftest>
		static size_t get_n_m(
			PVQuadTree const& obj,
			const uint64_t y1_min, const uint64_t y1_max,
			const uint64_t y2_min, const uint64_t y2_max,
			const uint32_t current_zoom,
			const double alpha,
			const Ftest &test_f, const insert_entry_y1_y2_f &insert_f,
			PVCore::PVHSVColor* image)
		{
			size_t ret = 0;
			if (obj._nodes != 0) {
				if (current_zoom == 0) {
					assert(obj._index_min_bg != PVROW_INVALID_VALUE);
					const PVQuadTreeEntry e(obj._y1_mid_value, obj._y2_mid_value, obj._index_min_bg);
					insert_f(e, image);
					return 1;
				}

				/* recursive search can be processed
				 */
				if (obj._y1_mid_value < y1_max) {
					if (obj._y2_mid_value < y2_max) {
						ret += get_n_m(obj._nodes[NE], y1_min, y1_max, y2_min, y2_max,
						current_zoom - 1, alpha,
						test_f, insert_f, image);
					}
					if (y2_min < obj._y2_mid_value) {
						ret += get_n_m(obj._nodes[SE], y1_min, y1_max, y2_min, y2_max,
						current_zoom - 1, alpha,
						test_f, insert_f, image);
					}
				}
				if (y1_min < obj._y1_mid_value) {
					if (obj._y2_mid_value < y2_max) {
						ret += get_n_m(obj._nodes[NW], y1_min, y1_max, y2_min, y2_max,
						current_zoom - 1, alpha,
						test_f, insert_f, image);
					}
					if (y2_min < obj._y2_mid_value) {
						ret += get_n_m(obj._nodes[SW], y1_min, y1_max, y2_min, y2_max,
						current_zoom - 1, alpha,
						test_f, insert_f, image);
					}
				}
			} else if (obj._datas.size() != 0) {
				/* this is a unsplitted node with data and an array of nxm
				 * entries is needed
				 */
				ret += extract_seq(obj, y1_min, y1_max, y2_min, y2_max, current_zoom, alpha,
							test_f, insert_f, image);
			}
			return ret;
		}

		/**
		 * Sequential function to extract a NxM events grid from a non-splitted quadtree.
		 *
		 * @param obj the quadtree to visit
		 * @param y1_min the included minimal bound along the y1 coordinate
		 * @param y1_max the excluded maximal bound along the y1 coordinate
		 * @param y2_min the included minimal bound along the y2 coordinate
		 * @param y2_max the excluded maximal bound along the y2 coordinate
		 * @param zoom the zoom level
		 * @param test_f the function executed to test if an event is relevant or not
		 * @param insert_f the function executed for each relevant found event
		 * @param image the output PVHSVColor image
		 */
		template <typename Ftest>
		static size_t extract_seq(PVQuadTree const& obj,
			const uint64_t y1_min, const uint64_t y1_max,
			const uint64_t y2_min, const uint64_t y2_max,
			const uint32_t current_zoom,
			const double alpha,
			const Ftest &test_f,
			const insert_entry_y1_y2_f &insert_f,
			PVCore::PVHSVColor* image)
		{
			size_t ret = 0;
			uint32_t img_width = ceil(double(1U << (current_zoom+1)) * alpha);
			uint32_t max_extraction_count = img_width*img_width;

			for(size_t i = 0; i < obj._datas.size(); ++i) {
				const PVQuadTreeEntry &e = obj._datas.at(i);
				if (!test_f(e, y1_min, y1_max, y2_min, y2_max)) {
					continue;
				}
				ret += insert_f(e, image);
				if (ret == max_extraction_count) {
					break;
				}
			}
			return ret;
		}

		template <typename Ftest>
		static size_t get_n_m_sel(
			PVQuadTree const& obj,
			const uint64_t y1_min, const uint64_t y1_max,
			const uint64_t y2_min, const uint64_t y2_max,
			const uint32_t current_zoom,
			const double alpha,
			const Ftest &test_f, const insert_entry_y1_y2_f &insert_f,
			PVCore::PVHSVColor* image,
			Picviz::PVSelection const& sel)
		{
			size_t ret = 0;
			if (obj._nodes != 0) {
				if (current_zoom == 0) {
					const PVRow idx = obj._index_min_sel;
					if (idx == PVROW_INVALID_VALUE) {
						return 0;
					}
					const PVQuadTreeEntry e(obj._y1_mid_value, obj._y2_mid_value, idx);
					insert_f(e, image);
					return 1;
				}

				/* recursive search can be processed
				 */
				if (obj._y1_mid_value < y1_max) {
					if (obj._y2_mid_value < y2_max) {
						ret += get_n_m(obj._nodes[NE], y1_min, y1_max, y2_min, y2_max,
						current_zoom - 1, alpha,
						test_f, insert_f, image);
					}
					if (y2_min < obj._y2_mid_value) {
						ret += get_n_m(obj._nodes[SE], y1_min, y1_max, y2_min, y2_max,
						current_zoom - 1, alpha,
						test_f, insert_f, image);
					}
				}
				if (y1_min < obj._y1_mid_value) {
					if (obj._y2_mid_value < y2_max) {
						ret += get_n_m(obj._nodes[NW], y1_min, y1_max, y2_min, y2_max,
						current_zoom - 1, alpha,
						test_f, insert_f, image);
					}
					if (y2_min < obj._y2_mid_value) {
						ret += get_n_m(obj._nodes[SW], y1_min, y1_max, y2_min, y2_max,
						current_zoom - 1, alpha,
						test_f, insert_f, image);
					}
				}
			} else if (obj._datas.size() != 0) {
				/* this is a unsplitted node with data and an array of nxm
				 * entries is needed
				 */
				ret += extract_seq_sel(obj, y1_min, y1_max, y2_min, y2_max, current_zoom, alpha,
							test_f, insert_f, image, sel);
			}
			return ret;
		}

		template <typename Ftest>
		static size_t extract_seq_sel(PVQuadTree const& obj,
			const uint64_t y1_min, const uint64_t y1_max,
			const uint64_t y2_min, const uint64_t y2_max,
			const uint32_t current_zoom,
			const double alpha,
			const Ftest &test_f,
			const insert_entry_y1_y2_f &insert_f,
			PVCore::PVHSVColor* image,
			Picviz::PVSelection const& sel)
		{
			size_t ret = 0;
			uint32_t img_width = ceil(double(1U << (current_zoom+1)) * alpha);
			uint32_t max_extraction_count = img_width*img_width;

			for(size_t i = 0; i < obj._datas.size(); ++i) {
				const PVQuadTreeEntry &e = obj._datas.at(i);
				if ((!sel.get_line_fast(e.idx)) || (!test_f(e, y1_min, y1_max, y2_min, y2_max))) {
					continue;
				}
				ret += insert_f(e, image);
				if (ret == max_extraction_count) {
					break;
				}
			}
			return ret;
		}
	};

	/**
	 * Traverse a quadtree to mark as selected all events in the range [y1_min,y1_max); there is no
	 * constraint on the y2 coordinate.
	 *
	 * @param obj the quadtree to visit
	 * @param y1_min the incluse minimal bound along the y1 coordinate
	 * @param y1_max the excluse maximal bound along the y1 coordinate
	 * @param selection the structure containing the result
	 *
	 * @return the count of selected events in selection
	 */
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

	/**
	 * Traverse a quadtree to mark as selected all events in the range [y2_min,y2_max); there is no
	 * constraint on the y1 coordinate.
	 *
	 * @param obj the quadtree to visit
	 * @param y2_min the incluse minimal bound along the y2 coordinate
	 * @param y2_max the excluse maximal bound along the y2 coordinate
	 * @param selection the structure containing the result
	 *
	 * @return the count of selected events in selection
	 */
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

	int32_t               _max_level;
	uint32_t			  _index_min_bg;
	uint32_t			  _index_min_sel;
};

#undef SW
#undef SE
#undef NW
#undef NE

}

#endif // PARALLELVIEW_PVQUADTREE_H
