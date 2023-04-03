//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <iostream>

#include <omp.h>

#include <pvhwloc.h>
#include <pvkernel/core/PVAlgorithms.h>
#include <pvkernel/core/PVUtils.h>

#include <pvparallelview/PVZoomedZoneTree.h>

#include <tbb/parallel_for.h>
#include <tbb/task_scheduler_init.h>

#define ZZT_MAX_VALUE (1 << (32 - NBITS_INDEX))

#define SEC_COORD_COUNT 2048

void merge_tlr(PVParallelView::PVZoomedZoneTree::context_t::tls_set_t& tls)
{
#if defined __SSE2__

	PVParallelView::PVZoomedZoneTree::context_t::tls_set_t::iterator it = tls.begin();
	uint32_t* res_data = it->get_tlr_buffer().get_data();
	++it;
	PVParallelView::PVZoomedZoneTree::context_t::tls_set_t::iterator it2 = it;

	size_t size = PVParallelView::PVZoomedZoneTree::pv_tlr_buffer_t::length;
	size_t packed_size = size & ~3;
	size_t i;

/* it the present case, iterating over each tlr buffer is faster than iterating
 * accross tlr (on sandy: 10 times faster)
 */
#if 1
	for (; it2 != tls.end(); ++it2) {
		uint32_t* val_data = it2->get_tlr_buffer().get_data();

		for (i = 0; i < packed_size; i += 4) {
			__m128i res_sse = _mm_loadu_si128((const __m128i*)&res_data[i]);
			__m128i val_sse = _mm_loadu_si128((const __m128i*)&(val_data[i]));
			res_sse = _mm_min_epu32(res_sse, val_sse);
			_mm_storeu_si128((__m128i*)&res_data[i], res_sse);
		}

		for (; i < size; ++i) {
			if (val_data[i] < res_data[i]) {
				res_data[i] = val_data[i];
			}
		}
	}
#else
	for (i = 0; i < packed_size; i += 4) {
		__m128i res_sse = _mm_loadu_si128((const __m128i*)&res_data[i]);

		for (it2 = it; it2 != tls.end(); ++it2) {
			uint32_t* val_data = it2->get_tlr_buffer().get_data();
			__m128i val_sse = _mm_loadu_si128((const __m128i*)&(val_data[i]));
			res_sse = _mm_min_epu32(res_sse, val_sse);
		}

		_mm_storeu_si128((__m128i*)&res_data[i], res_sse);
	}

	for (i = 0; i < packed_size; i += 4) {
		uint32_t res = res_data[i];

		for (it2 = it; it2 != tls.end(); ++it2) {
			uint32_t* val_data = it2->get_tlr_buffer().get_data();
			uint32_t val = val_data[i];
			if (val < res) {
				res = val;
			}
		}
		res_data[i] = res;
	}
#endif

#else
	PVParallelView::PVZoomedZoneTree::context_t::tls_set_t::iterator it = tls.begin();
	PVParallelView::PVZoomedZoneTree::pv_tlr_buffer_t& result = it->get_tlr_buffer();
	++it;

	for (; it != tls.end(); ++it) {
		PVParallelView::PVZoomedZoneTree::pv_tlr_buffer_t& tlr_buffer = it->get_tlr_buffer();

		for (size_t i = 0; i < PVParallelView::PVZoomedZoneTree::pv_tlr_buffer_t::length; ++i) {
			if (tlr_buffer[i] < result[i]) {
				result[i] = tlr_buffer[i];
			}
		}

		tlr_buffer.clear();
	}
#endif
}

/*****************************************************************************
 * compute_bci_projection_y1
 *****************************************************************************/

static inline void compute_bci_projection_y1(const uint64_t y1,
                                             const uint64_t y2,
                                             const uint64_t y_min,
                                             const uint64_t y_lim,
                                             const int shift,
                                             const uint32_t mask,
                                             const uint32_t width,
                                             const float beta,
                                             PVParallelView::PVZoomedZoneTree::pv_bci_code_t& bci)
{
	if (shift < 0) {
		bci.s.l = ((y2 - y_min) & mask) << -shift;
	} else {
		bci.s.l = ((y2 - y_min) >> shift) & mask;
	}

	int64_t d = (int64_t)y1 - (int64_t)y2;
	double y1p = (double)y2 + d * (double)beta;

	if (y1p >= y_lim) {
		bci.s.type = PVParallelView::PVZoomedZoneTree::pv_bci_code_t::DOWN;
		bci.s.r = ((double)width * (double)(y_lim - y2)) / (double)(y1p - y2);
	} else if (y1p < y_min) {
		bci.s.type = PVParallelView::PVZoomedZoneTree::pv_bci_code_t::UP;
		bci.s.r = ((double)width * (double)(y2 - y_min)) / (double)(y2 - y1p);
	} else {
		bci.s.type = PVParallelView::PVZoomedZoneTree::pv_bci_code_t::STRAIGHT;
		if (shift < 0) {
			bci.s.r = (((uint32_t)(y1p - y_min)) & mask) << -shift;
		} else {
			bci.s.r = (((uint32_t)(y1p - y_min)) >> shift) & mask;
		}
	}
}

/*****************************************************************************
 * compute_bci_projection_y2
 *****************************************************************************/

static inline void compute_bci_projection_y2(const uint64_t y1,
                                             const uint64_t y2,
                                             const uint64_t y_min,
                                             const uint64_t y_lim,
                                             const int shift,
                                             const uint32_t mask,
                                             const uint32_t width,
                                             const float beta,
                                             PVParallelView::PVZoomedZoneTree::pv_bci_code_t& bci)
{
	if (shift < 0) {
		bci.s.l = ((y1 - y_min) & mask) << -shift;
	} else {
		bci.s.l = ((y1 - y_min) >> shift) & mask;
	}

	int64_t dy = (int64_t)y2 - (int64_t)y1;
	double y2p = (double)y1 + dy * (double)beta;

	if (y2p >= y_lim) {
		bci.s.type = PVParallelView::PVZoomedZoneTree::pv_bci_code_t::DOWN;
		bci.s.r = ((double)width * (double)(y_lim - y1)) / (double)(y2p - y1);
	} else if (y2p < y_min) {
		bci.s.type = PVParallelView::PVZoomedZoneTree::pv_bci_code_t::UP;
		bci.s.r = ((double)width * (double)(y1 - y_min)) / (double)(y1 - y2p);
	} else {
		bci.s.type = PVParallelView::PVZoomedZoneTree::pv_bci_code_t::STRAIGHT;
		if (shift < 0) {
			bci.s.r = (((uint32_t)(y2p - y_min)) & mask) << -shift;
		} else {
			bci.s.r = (((uint32_t)(y2p - y_min)) >> shift) & mask;
		}
	}
}

/*****************************************************************************
 * compute_sec_coord_count_y1
 *****************************************************************************/

static inline uint32_t compute_sec_coord_count_y1(const uint32_t t1,
                                                  const uint32_t t2,
                                                  const uint64_t y_min,
                                                  const uint64_t y_lim,
                                                  const int shift,
                                                  const uint32_t mask,
                                                  const int zoom,
                                                  const uint32_t width,
                                                  const float beta)
{
	using bci_code_t = PVParallelView::PVZoomedZoneTree::pv_bci_code_t;
	using constants = PVParallelView::PVZoomedZoneTree::zzt_constants;

	bci_code_t bci_min, bci_max;
	uint32_t y1_count;

	compute_bci_projection_y1((uint64_t)BUCKET_ELT_COUNT * t1, (uint64_t)BUCKET_ELT_COUNT * t2,
	                          y_min, y_lim, shift, mask, width, beta, bci_min);

	compute_bci_projection_y1((uint64_t)BUCKET_ELT_COUNT * (t1 + 1),
	                          (uint64_t)BUCKET_ELT_COUNT * (t2 + 1), y_min, y_lim, shift, mask,
	                          width, beta, bci_max);

	if (bci_max.s.type == bci_code_t::UP) {
		// whole top side
		y1_count = PVCore::upper_power_of_2(bci_max.s.r - bci_min.s.r);
	} else if (bci_min.s.type == bci_code_t::DOWN) {
		// whole bottom side
		y1_count = PVCore::upper_power_of_2(bci_min.s.r - bci_max.s.r);
	} else if ((bci_min.s.type == bci_code_t::STRAIGHT) &&
	           (bci_max.s.type == bci_code_t::STRAIGHT)) {
		// opposite side
		y1_count = 1U << PVCore::clamp(zoom, 0, PARALLELVIEW_ZZT_BBITS);
	} else if (bci_min.s.type == bci_code_t::STRAIGHT) {
		// partial bottom side

		// opposite side
		y1_count = constants::image_height - bci_max.s.r;

		// + bottom side count
		y1_count += PARALLELVIEW_ZOOM_WIDTH - bci_min.s.r;

		y1_count = PVCore::upper_power_of_2(y1_count);
	} else if (bci_max.s.type == bci_code_t::STRAIGHT) {
		// partial top side

		// opposite side count
		y1_count = bci_min.s.r;

		// + top side count
		y1_count += PARALLELVIEW_ZOOM_WIDTH - bci_max.s.r;

		y1_count = PVCore::upper_power_of_2(y1_count);
	} else {
		// from top side to bottom side

		// opposite side count
		y1_count = constants::image_height;

		// + bottom side count
		y1_count += PARALLELVIEW_ZOOM_WIDTH - bci_max.s.r;

		// + top side count
		y1_count += PARALLELVIEW_ZOOM_WIDTH - bci_min.s.r;

		y1_count = PVCore::upper_power_of_2(y1_count);
	}

	return std::max(1U, y1_count);
}

/*****************************************************************************
 * compute_sec_coord_count_y2
 *****************************************************************************/

static inline uint32_t compute_sec_coord_count_y2(const uint32_t t1,
                                                  const uint32_t t2,
                                                  const uint64_t y_min,
                                                  const uint64_t y_lim,
                                                  const int shift,
                                                  const uint32_t mask,
                                                  const int zoom,
                                                  const uint32_t width,
                                                  const float beta)
{
	using bci_code_t = PVParallelView::PVZoomedZoneTree::pv_bci_code_t;
	using constants = PVParallelView::PVZoomedZoneTree::zzt_constants;

	bci_code_t bci_min, bci_max;
	uint32_t y2_count;

	compute_bci_projection_y2((uint64_t)BUCKET_ELT_COUNT * t1, (uint64_t)BUCKET_ELT_COUNT * t2,
	                          y_min, y_lim, shift, mask, width, beta, bci_min);

	compute_bci_projection_y2((uint64_t)BUCKET_ELT_COUNT * (t1 + 1),
	                          (uint64_t)BUCKET_ELT_COUNT * (t2 + 1), y_min, y_lim, shift, mask,
	                          width, beta, bci_max);

	if (bci_max.s.type == bci_code_t::UP) {
		// whole top side
		y2_count = PVCore::upper_power_of_2(bci_max.s.r - bci_min.s.r);
	} else if (bci_min.s.type == bci_code_t::DOWN) {
		// whole bottom side
		y2_count = PVCore::upper_power_of_2(bci_min.s.r - bci_max.s.r);
	} else if ((bci_min.s.type == bci_code_t::STRAIGHT) &&
	           (bci_max.s.type == bci_code_t::STRAIGHT)) {
		// opposite side
		y2_count = 1U << PVCore::clamp(zoom, 0, PARALLELVIEW_ZZT_BBITS);
	} else if (bci_min.s.type == bci_code_t::STRAIGHT) {
		// partial bottom side

		// opposite side
		y2_count = constants::image_height - bci_max.s.r;

		// + bottom side count
		y2_count += PARALLELVIEW_ZOOM_WIDTH - bci_min.s.r;

		y2_count = PVCore::upper_power_of_2(y2_count);
	} else if (bci_max.s.type == bci_code_t::STRAIGHT) {
		// partial top side

		// opposite side count
		y2_count = bci_min.s.r;

		// + top side count
		y2_count += PARALLELVIEW_ZOOM_WIDTH - bci_max.s.r;

		y2_count = PVCore::upper_power_of_2(y2_count);
	} else {
		// from top side to bottom side

		// opposite side count
		y2_count = constants::image_height;

		// + bottom side count
		y2_count += PARALLELVIEW_ZOOM_WIDTH - bci_max.s.r;

		// + top side count
		y2_count += PARALLELVIEW_ZOOM_WIDTH - bci_min.s.r;

		y2_count = PVCore::upper_power_of_2(y2_count);
	}

	return std::max(1U, y2_count);
}

/*****************************************************************************
 * PVParallelView::PVZoomedZoneTree::PVZoomedZoneTree
 *****************************************************************************/

PVParallelView::PVZoomedZoneTree::PVZoomedZoneTree(const PVRow* sel_elts,
                                                   const PVRow* bg_elts,
                                                   uint32_t max_level)
    : _trees(nullptr)
    , _sel_elts(sel_elts)
    , _bg_elts(bg_elts)
    , _max_level(max_level)
    , _initialized(false)
{
}

/*****************************************************************************
 * PVParallelView::PVZoomedZoneTree::~PVZoomedZoneTree
 *****************************************************************************/

PVParallelView::PVZoomedZoneTree::~PVZoomedZoneTree()
{
	reset();
}

/*****************************************************************************
 * PVParallelView::PVZoomedZoneTree::reset
 *****************************************************************************/

void PVParallelView::PVZoomedZoneTree::reset()
{
	if (_trees != nullptr) {
		delete[] _trees;
		_trees = nullptr;
	}
	_initialized = false;
}

void PVParallelView::PVZoomedZoneTree::init_structures()
{
	if (_trees) {
		return;
	}

	uint32_t idx = 0;
	uint32_t y1_min;
	uint32_t y2_min;

	_trees = new pvquadtree[NBUCKETS];
	y2_min = 0;
	for (uint32_t y2 = 0; y2 < 1024; ++y2) {
		y1_min = 0;
		for (uint32_t y1 = 0; y1 < 1024; ++y1) {
			_trees[idx].init(y1_min, y1_min + (ZZT_MAX_VALUE >> 1), y2_min,
			                 y2_min + (ZZT_MAX_VALUE >> 1), _max_level);
			y1_min += ZZT_MAX_VALUE;
			++idx;
		}
		y2_min += ZZT_MAX_VALUE;
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedZoneTree::process
 *****************************************************************************/

void PVParallelView::PVZoomedZoneTree::process(const PVZoneProcessing& zp, PVZoneTree& zt)
{
	if (_initialized) {
		PVLOG_WARN("calling ::process() on an already initialized ZoomedZoneTree\n");
		return;
	}

	init_structures();

#ifdef INENDI_DEVELOPER_MODE
	tbb::tick_count start, end;
	start = tbb::tick_count::now();
#endif
	process_omp_from_zt(zp, zt);

#ifdef INENDI_DEVELOPER_MODE
	end = tbb::tick_count::now();
	PVLOG_INFO("PVZoomedZoneTree::process in %0.4f ms.\n", (end - start).seconds() * 1000.0);
	PVLOG_INFO("PVZoomedZoneTree::memory: %lu octets.\n", memory());
#endif
	_initialized = true;
}

/*****************************************************************************
 * PVParallelView::PVZoomedZoneTree::process_omp_from_zt
 *****************************************************************************/

void PVParallelView::PVZoomedZoneTree::process_omp_from_zt(const PVZoneProcessing& zp,
                                                           PVZoneTree& zt)
{
	init_structures();

	const uint32_t* pcol_a = zp.plotted_a;
	const uint32_t* pcol_b = zp.plotted_b;

	BENCH_START(zztree);
	tbb::parallel_for(
	    tbb::blocked_range<size_t>(0, NBUCKETS, 256),
	    [&](tbb::blocked_range<size_t> const& range) {
		    pvquadtree* trees = this->_trees;
		    PVZoneTree& zt_(zt);
		    const uint32_t* pcol_a_ = pcol_a;
		    const uint32_t* pcol_b_ = pcol_b;
		    for (size_t i = range.begin(); i != range.end(); i++) {
			    pvquadtree& tree_i = trees[i];
			    for (size_t j = 0; j < zt_.get_branch_count(i); ++j) {
				    const PVRow r = zt_.get_branch_element(i, j);

				    PVParallelView::PVQuadTreeEntry e(pcol_a_[r], pcol_b_[r], r);
				    tree_i.insert(e);
			    }
			    tree_i.compact();
		    }
	    },
	    tbb::auto_partitioner());
	BENCH_END(zztree, "ZZTREE CREATION (PARALLEL)", 1, 1, 1, 1);
}

/*****************************************************************************
 * PVParallelView::PVZoomedZoneTree::browse_trees_bci_by_y1
 *****************************************************************************/

size_t
PVParallelView::PVZoomedZoneTree::browse_trees_bci_by_y1_seq(context_t& ctx,
                                                             uint64_t y_min,
                                                             uint64_t y_max,
                                                             uint64_t y_lim,
                                                             int zoom,
                                                             uint32_t width,
                                                             const extract_entries_f& extract_f,
                                                             const PVCore::PVHSVColor* colors,
                                                             pv_bci_code_t* codes,
                                                             const float beta,
                                                             const PVRow* sel_elts) const
{
	uint32_t shift = (32 - bbits) - zoom;
	uint32_t t1_min = y_min >> (32 - NBITS_INDEX);
	uint32_t t1_max =
	    (uint32_t)PVCore::clamp<uint64_t>(1 + (y_max >> (32 - NBITS_INDEX)), 0U, 1024U);
	zzt_tls& tls = ctx.get_tls().local();
	pv_tlr_buffer_t& tlr_buffer = tls.get_tlr_buffer();
	pv_quadtree_buffer_entry_t* quadtree_buffer = tls.get_quadtree_buffer();

	const insert_entry_f insert_f =
	    insert_entry_f([&](const PVQuadTreeEntry& e, pv_tlr_buffer_t& buffer) {
		    pv_bci_code_t bci;
		    compute_bci_projection_y2(e.y1, e.y2, y_min, y_lim, shift, mask_int_ycoord, width, beta,
		                              bci);
		    pv_tlr_index_t tlr(bci.s.type, bci.s.l, bci.s.r);
		    if (e.idx < buffer[tlr.v]) {
			    buffer[tlr.v] = e.idx;
		    }
	    });

	BENCH_START(whole);
	BENCH_START(extract);
	for (uint32_t t1 = t1_min; t1 < t1_max; ++t1) {
		for (uint32_t t2 = 0; t2 < 1024; ++t2) {
			PVRow tree_idx = (t2 * 1024) + t1;

			if (sel_elts && (sel_elts[tree_idx] == PVROW_INVALID_VALUE)) {
				/* when searching for entries using the selection, if there is no
				 * drawn selected event for the corresponding ZoneTree, it is useless
				 * to search for a selected event in the quadtree
				 */
				continue;
			}

			/* compute the events number along the secondary coordinate
			 */
			const uint32_t y2_count = compute_sec_coord_count_y2(
			    t1, t2, y_min, y_lim, shift, mask_int_ycoord, zoom, width, beta);

			/* lines extraction
			 */
			extract_f(_trees[tree_idx], y2_count, quadtree_buffer, tlr_buffer, insert_f);
		}
	}
	BENCH_STOP(extract);

	/* extracting BCI codes from TLR buffer
	 */
	size_t bci_idx = 0;

	BENCH_START(compute);
	for (size_t i = 0; i < pv_tlr_buffer_t::length; ++i) {
		const pv_tlr_index_t tlr(i);

		const uint32_t idx = tlr_buffer[i];
		if (idx != UINT32_MAX) {
			pv_bci_code_t& bci = codes[bci_idx];
			++bci_idx;
			bci.s.idx = idx;
			bci.s.type = tlr.s.t;
			bci.s.l = tlr.s.l;
			bci.s.r = tlr.s.r;
			bci.s.color = colors[idx].h();
		}
	}
	BENCH_STOP(compute);
	BENCH_STOP(whole);

	tlr_buffer.clear();

	if (sel_elts) {
		BENCH_SHOW(extract, "extraction  y1 seq sel", 1, 1, 1, 1);
		BENCH_SHOW(compute, "computation y1 seq sel", 1, 1, 1, 1);
		BENCH_SHOW(whole, "whole       y1 seq sel", 1, 1, 1, 1);
	} else {
		BENCH_SHOW(extract, "extraction  y1 seq bg", 1, 1, 1, 1);
		BENCH_SHOW(compute, "computation y1 seq bg", 1, 1, 1, 1);
		BENCH_SHOW(whole, "whole       y1 seq bg", 1, 1, 1, 1);
	}

	PVLOG_DEBUG("::browse_trees_bci_by_y1_seq -> %lu\n", bci_idx);

	return bci_idx;
}

/*****************************************************************************
 * PVParallelView::PVZoomedZoneTree::browse_trees_bci_by_y2
 *****************************************************************************/

size_t
PVParallelView::PVZoomedZoneTree::browse_trees_bci_by_y2_seq(context_t& ctx,
                                                             uint64_t y_min,
                                                             uint64_t y_max,
                                                             uint64_t y_lim,
                                                             int zoom,
                                                             uint32_t width,
                                                             const extract_entries_f& extract_f,
                                                             const PVCore::PVHSVColor* colors,
                                                             pv_bci_code_t* codes,
                                                             const float beta,
                                                             const PVRow* sel_elts) const
{
	uint32_t shift = (32 - bbits) - zoom;
	uint32_t t2_min = y_min >> (32 - NBITS_INDEX);
	uint32_t t2_max =
	    (uint32_t)PVCore::clamp<uint64_t>(1 + (y_max >> (32 - NBITS_INDEX)), 0U, 1024U);
	zzt_tls& tls = ctx.get_tls().local();
	pv_tlr_buffer_t& tlr_buffer = tls.get_tlr_buffer();
	pv_quadtree_buffer_entry_t* quadtree_buffer = tls.get_quadtree_buffer();

	const insert_entry_f insert_f =
	    insert_entry_f([&](const PVQuadTreeEntry& e, pv_tlr_buffer_t& buffer) {
		    pv_bci_code_t bci;
		    compute_bci_projection_y1(e.y1, e.y2, y_min, y_lim, shift, mask_int_ycoord, width, beta,
		                              bci);
		    pv_tlr_index_t tlr(bci.s.type, bci.s.l, bci.s.r);
		    if (e.idx < buffer[tlr.v]) {
			    buffer[tlr.v] = e.idx;
		    }
	    });

	BENCH_START(whole);
	BENCH_START(extract);
	for (uint32_t t2 = t2_min; t2 < t2_max; ++t2) {
		for (uint32_t t1 = 0; t1 < 1024; ++t1) {
			PVRow tree_idx = (t2 * 1024) + t1;

			if (sel_elts && (sel_elts[tree_idx] == PVROW_INVALID_VALUE)) {
				/* when searching for entries using the selection, if there is no
				 * drawn selected event for the corresponding ZoneTree, it is useless
				 * to search for a selected event in the quadtree
				 */
				continue;
			}

			/* compute the events number along the secondary coordinate
			 */
			const uint32_t y1_count = compute_sec_coord_count_y1(
			    t1, t2, y_min, y_lim, shift, mask_int_ycoord, zoom, width, beta);

			/* lines extraction
			 */
			extract_f(_trees[tree_idx], y1_count, quadtree_buffer, tlr_buffer, insert_f);
		}
	}
	BENCH_STOP(extract);

	/* extracting BCI codes from TLR buffer
	 */
	size_t bci_idx = 0;

	BENCH_START(compute);
	for (size_t i = 0; i < pv_tlr_buffer_t::length; ++i) {
		const pv_tlr_index_t tlr(i);

		const uint32_t idx = tlr_buffer[i];
		if (idx != UINT32_MAX) {
			pv_bci_code_t& bci = codes[bci_idx];
			++bci_idx;
			bci.s.idx = idx;
			bci.s.type = tlr.s.t;
			bci.s.l = tlr.s.l;
			bci.s.r = tlr.s.r;
			bci.s.color = colors[idx].h();
		}
	}
	BENCH_STOP(compute);
	BENCH_STOP(whole);

	tlr_buffer.clear();

	if (sel_elts) {
		BENCH_SHOW(extract, "extraction  y2 seq sel", 1, 1, 1, 1);
		BENCH_SHOW(compute, "computation y2 seq sel", 1, 1, 1, 1);
		BENCH_SHOW(whole, "whole       y2 seq sel", 1, 1, 1, 1);
	} else {
		BENCH_SHOW(extract, "extraction  y2 seq bg", 1, 1, 1, 1);
		BENCH_SHOW(compute, "computation y2 seq bg", 1, 1, 1, 1);
		BENCH_SHOW(whole, "whole       y2 seq sel", 1, 1, 1, 1);
	}

	PVLOG_DEBUG("::browse_trees_bci_by_y2_seq -> %lu\n", bci_idx);

	return bci_idx;
}

void PVParallelView::PVZoomedZoneTree::browse_trees_bci_by_y1_y2_tbb(
    uint64_t y1_min,
    uint64_t y1_max,
    uint64_t y2_min,
    uint64_t y2_max,
    int zoom,
    double alpha,
    PVCore::PVHSVColor const* const colors,
    PVCore::PVHSVColor* const image,
    uint32_t image_width,
    const extract_entries_y1_y2_f& extract_f,
    PVRow const* const sel_elts,
    tbb::task_group_context* tbb_ctxt) const
{
	tbb::task_group_context my_ctxt;
	if (tbb_ctxt == nullptr) {
		tbb_ctxt = &my_ctxt;
	} else if (tbb_ctxt->is_group_execution_cancelled()) {
		return;
	}

	uint32_t shift = (32 - PARALLELVIEW_ZT_BBITS) - zoom;
	uint32_t t1_min = y1_min >> (32 - NBITS_INDEX);
	uint32_t t1_max =
	    (uint32_t)PVCore::clamp<uint64_t>(1 + (y1_max >> (32 - NBITS_INDEX)), 0U, 1024U);

	uint32_t t2_min = y2_min >> (32 - NBITS_INDEX);
	uint32_t t2_max =
	    (uint32_t)PVCore::clamp<uint64_t>(1 + (y2_max >> (32 - NBITS_INDEX)), 0U, 1024U);

	const insert_entry_y1_y2_f insert_f =
	    insert_entry_y1_y2_f([=](const PVQuadTreeEntry& e, PVCore::PVHSVColor* image) {
		    /**
		     * RH: the 2 following tests must never occur but they
		     * do... As the bug occurs in a 2 levels tbb::task mess, it
		     * is really really really really hard to debug properly (I
		     * don't arrive to know if the problem comes from tbb or
		     * from the quadtree (in the fill or in extraction)).
		     *
		     * The "fast" way is to reject "bad" events...
		     */
		    if (y1_min > e.y1) {
			    return 0;
		    }

		    if (y2_min > e.y2) {
			    return 0;
		    }

		    uint32_t l = ((uint32_t)(((e.y1 - y1_min) * alpha))) >> shift;
		    uint32_t r = ((uint32_t)(((e.y2 - y2_min) * alpha))) >> shift;

		    assert(r < image_width);
		    assert(l < image_width);

		    if (image[r * image_width + l] == HSV_COLOR_TRANSPARENT) {
			    image[r * image_width + l] = colors[e.idx];
			    return 1;
		    }
		    return 0;
	    });

	BENCH_START(extract);
	tbb::parallel_for(
	    tbb::blocked_range2d<uint32_t>(t1_min, t1_max, t2_min, t2_max),
	    [&](const tbb::blocked_range2d<uint32_t>& r) {
		    pvquadtree* trees = _trees;
		    const insert_entry_y1_y2_f& insert_f_ = insert_f;
		    PVCore::PVHSVColor* const image_ = image;
		    PVRow const* const sel_elts_ = sel_elts;

		    for (uint32_t t2 = r.cols().begin(); t2 < r.cols().end(); ++t2) {
			    for (uint32_t t1 = r.rows().begin(); t1 < r.rows().end(); ++t1) {
				    PVRow tree_idx = (t2 * 1024) + t1;

				    if (sel_elts_ && (sel_elts_[tree_idx] == PVROW_INVALID_VALUE)) {
					    /* when searching for entries using the selection, if there is
					     * no
					     * drawn selected event for the corresponding ZoneTree, it is
					     * useless
					     * to search for a selected event in the quadtree
					     */
					    continue;
				    }

				    /* lines extraction
				     */
				    extract_f(trees[tree_idx], image_, insert_f_);
			    }
		    }
	    },
	    tbb::auto_partitioner(), *tbb_ctxt);
	BENCH_END(extract, "browse_trees_bci_by_y1_y2_tbb", 1, 1, 1, 1);
}

/*****************************************************************************
 * PVParallelView::PVZoomedZoneTree::browse_trees_bci_by_y1_tbb
 *****************************************************************************/

size_t
PVParallelView::PVZoomedZoneTree::browse_trees_bci_by_y1_tbb(context_t& ctx,
                                                             uint64_t y_min,
                                                             uint64_t y_max,
                                                             uint64_t y_lim,
                                                             int zoom,
                                                             uint32_t width,
                                                             const extract_entries_f& extract_f,
                                                             const PVCore::PVHSVColor* colors,
                                                             pv_bci_code_t* codes,
                                                             const float beta,
                                                             const PVRow* sel_elts) const
{
	uint32_t shift = (32 - bbits) - zoom;
	uint32_t t1_min = y_min >> (32 - NBITS_INDEX);
	uint32_t t1_max =
	    (uint32_t)PVCore::clamp<uint64_t>(1 + (y_max >> (32 - NBITS_INDEX)), 0U, 1024U);

	const insert_entry_f insert_f =
	    insert_entry_f([&](const PVQuadTreeEntry& e, pv_tlr_buffer_t& buffer) {
		    pv_bci_code_t bci;
		    compute_bci_projection_y2(e.y1, e.y2, y_min, y_lim, shift, mask_int_ycoord, width, beta,
		                              bci);
		    pv_tlr_index_t tlr(bci.s.type, bci.s.l, bci.s.r);
		    if (e.idx < buffer[tlr.v]) {
			    buffer[tlr.v] = e.idx;
		    }
	    });

	BENCH_START(whole);
	BENCH_START(extract);
	tbb::parallel_for(
	    tbb::blocked_range2d<uint32_t>(t1_min, t1_max, 0, 1024),
	    [&](const tbb::blocked_range2d<uint32_t>& r) {
		    zzt_tls& tls = ctx.get_tls().local();
		    pv_tlr_buffer_t& tlr_buffer = tls.get_tlr_buffer();
		    pv_quadtree_buffer_entry_t* quadtree_buffer = tls.get_quadtree_buffer();

		    // AG: copy this variable to the local stack (or better, register), which
		    // may reduce the number
		    // of loads from the original stack.
		    const PVRow* sel_elts_ = sel_elts;
		    for (uint32_t t1 = r.rows().begin(); t1 != r.rows().end(); ++t1) {
			    for (uint32_t t2 = r.cols().begin(); t2 != r.cols().end(); ++t2) {
				    PVRow tree_idx = (t2 * 1024) + t1;

				    if (sel_elts_ && (sel_elts_[tree_idx] == PVROW_INVALID_VALUE)) {
					    /* when searching for entries using the selection, if there is no
					     * drawn selected event for the corresponding ZoneTree, it is useless
					     * to search for a selected event in the quadtree
					     */
					    continue;
				    }

				    /* compute the events number along the secondary coordinate
				     */
				    const uint32_t y2_count = compute_sec_coord_count_y2(
				        t1, t2, y_min, y_lim, shift, mask_int_ycoord, zoom, width, beta);

				    /* lines extraction
				     */
				    extract_f(_trees[tree_idx], y2_count, quadtree_buffer, tlr_buffer, insert_f);
			    }
		    }
	    });
	BENCH_STOP(extract);

	/* merging all TLR buffers
	 */
	BENCH_START(merge);
	merge_tlr(ctx.get_tls());
	BENCH_STOP(merge);
	pv_tlr_buffer_t& tlr_buffer = ctx.get_tls().begin()->get_tlr_buffer();

	/* extracting BCI codes from TLR buffer
	 */
	size_t bci_idx = 0;

	BENCH_START(compute);
	for (size_t i = 0; i < pv_tlr_buffer_t::length; ++i) {
		const pv_tlr_index_t tlr(i);

		const uint32_t idx = tlr_buffer[i];
		if (idx != UINT32_MAX) {
			pv_bci_code_t& bci = codes[bci_idx];
			++bci_idx;
			bci.s.idx = idx;
			bci.s.type = tlr.s.t;
			bci.s.l = tlr.s.l;
			bci.s.r = tlr.s.r;
			bci.s.color = colors[idx].h();
		}
	}
	BENCH_STOP(compute);
	BENCH_STOP(whole);

	tlr_buffer.clear();

	if (sel_elts) {
		BENCH_SHOW(extract, "extraction  y1 tbb sel", 1, 1, 1, 1);
		BENCH_SHOW(merge, "merge       y1 tbb sel", 1, 1, 1, 1);
		BENCH_SHOW(compute, "computation y1 tbb sel", 1, 1, 1, 1);
		BENCH_SHOW(whole, "whole       y1 tbb sel", 1, 1, 1, 1);
	} else {
		BENCH_SHOW(extract, "extraction  y1 tbb bg", 1, 1, 1, 1);
		BENCH_SHOW(merge, "merge       y1 tbb bg", 1, 1, 1, 1);
		BENCH_SHOW(compute, "computation y1 tbb bg", 1, 1, 1, 1);
		BENCH_SHOW(whole, "whole       y1 tbb bg", 1, 1, 1, 1);
	}

	PVLOG_DEBUG("::browse_trees_bci_by_y1_tbb -> %lu\n", bci_idx);

	return bci_idx;
}

/*****************************************************************************
 * PVParallelView::PVZoomedZoneTree::browse_trees_bci_by_y2_tbb
 *****************************************************************************/

size_t
PVParallelView::PVZoomedZoneTree::browse_trees_bci_by_y2_tbb(context_t& ctx,
                                                             uint64_t y_min,
                                                             uint64_t y_max,
                                                             uint64_t y_lim,
                                                             int zoom,
                                                             uint32_t width,
                                                             const extract_entries_f& extract_f,
                                                             const PVCore::PVHSVColor* colors,
                                                             pv_bci_code_t* codes,
                                                             const float beta,
                                                             const PVRow* sel_elts) const
{
	uint32_t shift = (32 - bbits) - zoom;
	uint32_t t2_min = y_min >> (32 - NBITS_INDEX);
	uint32_t t2_max =
	    (uint32_t)PVCore::clamp<uint64_t>(1 + (y_max >> (32 - NBITS_INDEX)), 0U, 1024U);

	const insert_entry_f insert_f =
	    insert_entry_f([&](const PVQuadTreeEntry& e, pv_tlr_buffer_t& buffer) {
		    pv_bci_code_t bci;
		    compute_bci_projection_y1(e.y1, e.y2, y_min, y_lim, shift, mask_int_ycoord, width, beta,
		                              bci);
		    pv_tlr_index_t tlr(bci.s.type, bci.s.l, bci.s.r);
		    if (e.idx < buffer[tlr.v]) {
			    buffer[tlr.v] = e.idx;
		    }
	    });

	BENCH_START(whole);
	BENCH_START(extract);
	tbb::parallel_for(
	    tbb::blocked_range2d<uint32_t>(t2_min, t2_max, 0, 1024),
	    [&](const tbb::blocked_range2d<uint32_t>& r) {
		    zzt_tls& tls = ctx.get_tls().local();
		    pv_tlr_buffer_t& tlr_buffer = tls.get_tlr_buffer();
		    pv_quadtree_buffer_entry_t* quadtree_buffer = tls.get_quadtree_buffer();

		    // AG: this copy is wanted. cf. browse_trees_bci_by_y1_tbb.
		    const PVRow* sel_elts_ = sel_elts;
		    for (uint32_t t2 = r.rows().begin(); t2 != r.rows().end(); ++t2) {
			    for (uint32_t t1 = r.cols().begin(); t1 != r.cols().end(); ++t1) {
				    PVRow tree_idx = (t2 * 1024) + t1;

				    if (sel_elts_ && (sel_elts_[tree_idx] == PVROW_INVALID_VALUE)) {
					    /* when searching for entries using the selection, if there is no
					     * drawn selected event for the corresponding ZoneTree, it is useless
					     * to search for a selected event in the quadtree
					     */
					    continue;
				    }

				    /* compute the events number along the secondary coordinate
				     */
				    const uint32_t y1_count = compute_sec_coord_count_y1(
				        t1, t2, y_min, y_lim, shift, mask_int_ycoord, zoom, width, beta);

				    /* lines extraction
				     */
				    extract_f(_trees[tree_idx], y1_count, quadtree_buffer, tlr_buffer, insert_f);
			    }
		    }
	    });
	BENCH_STOP(extract);

	/* merging all TLR buffers
	 */
	BENCH_START(merge);
	merge_tlr(ctx.get_tls());
	BENCH_STOP(merge);
	pv_tlr_buffer_t& tlr_buffer = ctx.get_tls().begin()->get_tlr_buffer();

	/* extracting BCI codes from TLR buffer
	 */
	size_t bci_idx = 0;

	BENCH_START(compute);
	for (size_t i = 0; i < pv_tlr_buffer_t::length; ++i) {
		const pv_tlr_index_t tlr(i);

		const uint32_t idx = tlr_buffer[i];
		if (idx != UINT32_MAX) {
			pv_bci_code_t& bci = codes[bci_idx];
			++bci_idx;
			bci.s.idx = idx;
			bci.s.type = tlr.s.t;
			bci.s.l = tlr.s.l;
			bci.s.r = tlr.s.r;
			bci.s.color = colors[idx].h();
		}
	}
	BENCH_STOP(compute);
	BENCH_STOP(whole);

	tlr_buffer.clear();

	if (sel_elts) {
		BENCH_SHOW(extract, "extraction  y2 tbb sel", 1, 1, 1, 1);
		BENCH_SHOW(merge, "merge       y2 tbb sel", 1, 1, 1, 1);
		BENCH_SHOW(compute, "computation y2 tbb sel", 1, 1, 1, 1);
		BENCH_SHOW(whole, "whole       y2 tbb sel", 1, 1, 1, 1);
	} else {
		BENCH_SHOW(extract, "extraction  y2 tbb bg", 1, 1, 1, 1);
		BENCH_SHOW(merge, "merge       y2 tbb bg", 1, 1, 1, 1);
		BENCH_SHOW(compute, "computation y2 tbb bg", 1, 1, 1, 1);
		BENCH_SHOW(whole, "whole       y2 tbb bg", 1, 1, 1, 1);
	}

	PVLOG_DEBUG("::browse_trees_bci_by_y2_tbb -> %lu\n", bci_idx);

	return bci_idx;
}

void PVParallelView::PVZoomedZoneTree::compute_min_indexes_sel(Inendi::PVSelection const& sel)
{
	BENCH_START(compute);
	if (sel.is_empty()) {
		for (size_t i = 0; i < NBUCKETS; i++) {
			_trees[i].set_min_idx_sel_invalid();
		}
	} else {
		PVRow const* zt_sel_elts = _sel_elts;

		for (size_t i = 0; i < NBUCKETS; i++) {
			if (zt_sel_elts[i] == PVROW_INVALID_VALUE) {
				_trees[i].set_min_idx_sel_invalid();
			} else {
				_trees[i].compute_min_indexes_sel_notempty(sel);
			}
		}
	}
	BENCH_END(compute, "PVZoomedZoneTree::compute_min_indexes_sel", 1, 1, 1, 1);
}

#ifdef INENDI_DEVELOPER_MODE
double PVParallelView::extract_stat::all_dt = 0;
size_t PVParallelView::extract_stat::all_cnt = 0;
size_t PVParallelView::extract_stat::test_cnt = 0;
size_t PVParallelView::extract_stat::insert_cnt = 0;
#endif
