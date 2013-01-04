/**
 * \file PVZoomedZoneTree.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <iostream>

#include <omp.h>

#include <pvkernel/core/PVHardwareConcurrency.h>
#include <pvkernel/core/PVAlgorithms.h>

#include <pvparallelview/PVZoomedZoneTree.h>

#include <tbb/parallel_for.h>
#include <tbb/task_scheduler_init.h>

#define ZZT_MAX_VALUE (1 << (32-NBITS_INDEX))

#define SEC_COORD_COUNT 2048

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
                                             PVParallelView::PVZoomedZoneTree::pv_bci_code_t &bci)
{
	if (shift < 0) {
		bci.s.l = ((y2 - y_min) & mask ) << -shift;
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
                                             PVParallelView::PVZoomedZoneTree::pv_bci_code_t &bci)
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
	typedef PVParallelView::PVZoomedZoneTree::pv_bci_code_t bci_code_t;
	typedef PVParallelView::PVZoomedZoneTree::zzt_constants constants;

	bci_code_t bci_min, bci_max;
	uint32_t y1_count;

	compute_bci_projection_y1((uint64_t)BUCKET_ELT_COUNT * t1,
	                          (uint64_t)BUCKET_ELT_COUNT * t2,
	                          y_min, y_lim,
	                          shift, mask,
	                          width, beta, bci_min);

	compute_bci_projection_y1((uint64_t)BUCKET_ELT_COUNT * (t1 + 1),
	                          (uint64_t)BUCKET_ELT_COUNT * (t2 + 1),
	                          y_min, y_lim,
	                          shift, mask,
	                          width, beta, bci_max);

	if (bci_max.s.type == bci_code_t::UP) {
		// whole top side
		y1_count = PVCore::upper_power_of_2(bci_max.s.r - bci_min.s.r);
	} else if (bci_min.s.type == bci_code_t::DOWN) {
		// whole bottom side
		y1_count = PVCore::upper_power_of_2(bci_min.s.r - bci_max.s.r);
	} else if ((bci_min.s.type == bci_code_t::STRAIGHT)
	           &&
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

	return PVCore::max(1U, y1_count);
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
	typedef PVParallelView::PVZoomedZoneTree::pv_bci_code_t bci_code_t;
	typedef PVParallelView::PVZoomedZoneTree::zzt_constants constants;

	bci_code_t bci_min, bci_max;
	uint32_t y2_count;

	compute_bci_projection_y2((uint64_t)BUCKET_ELT_COUNT * t1,
	                          (uint64_t)BUCKET_ELT_COUNT * t2,
	                          y_min, y_lim,
	                          shift, mask,
	                          width, beta, bci_min);

	compute_bci_projection_y2((uint64_t)BUCKET_ELT_COUNT * (t1 + 1),
	                          (uint64_t)BUCKET_ELT_COUNT * (t2 + 1),
	                          y_min, y_lim,
	                          shift, mask,
	                          width, beta, bci_max);

	if (bci_max.s.type == bci_code_t::UP) {
		// whole top side
		y2_count = PVCore::upper_power_of_2(bci_max.s.r - bci_min.s.r);
	} else if (bci_min.s.type == bci_code_t::DOWN) {
		// whole bottom side
		y2_count = PVCore::upper_power_of_2(bci_min.s.r - bci_max.s.r);
	} else if ((bci_min.s.type == bci_code_t::STRAIGHT)
	           &&
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

	return PVCore::max(1U, y2_count);
}

/*****************************************************************************
 * PVParallelView::PVZoomedZoneTree::PVZoomedZoneTree
 *****************************************************************************/

PVParallelView::PVZoomedZoneTree::PVZoomedZoneTree(const PVRow *sel_elts,
                                                   const PVRow *bg_elts,
                                                   uint32_t max_level) :
	_trees(nullptr),
	_sel_elts(sel_elts),
	_bg_elts(bg_elts),
	_max_level(max_level),
	_initialized(false)
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
		delete [] _trees;
		_trees = nullptr;
	}
	_initialized = false;
}

/*****************************************************************************
 * PVParallelView::PVZoomedZoneTree::process
 *****************************************************************************/

void PVParallelView::PVZoomedZoneTree::process(const PVZoneProcessing &zp,
                                               PVZoneTree &zt)
{
	if (_initialized) {
		PVLOG_WARN("calling ::process() on an already initialized ZoomedZoneTree\n");
		return;
	}

	uint32_t idx = 0;
	uint32_t y1_min;
	uint32_t y2_min;

	_trees = new pvquadtree [NBUCKETS];
	y2_min = 0;
	for(uint32_t y2 = 0; y2 < 1024; ++y2) {
		y1_min = 0;
		for(uint32_t y1 = 0; y1 < 1024; ++y1) {
			_trees[idx].init(y1_min, y1_min + (ZZT_MAX_VALUE >> 1),
			                 y2_min, y2_min + (ZZT_MAX_VALUE >> 1),
			                 _max_level);
			y1_min += ZZT_MAX_VALUE;
			++idx;
		}
		y2_min += ZZT_MAX_VALUE;
	}

	tbb::tick_count start, end;
	start = tbb::tick_count::now();
	process_omp_from_zt(zp, zt);
	end = tbb::tick_count::now();
	PVLOG_INFO("PVZoomedZoneTree::process in %0.4f ms.\n", (end-start).seconds()*1000.0);
	PVLOG_INFO("PVZoomedZoneTree::memory: %lu octets.\n", memory());
	_initialized = true;
}

/*****************************************************************************
 * PVParallelView::PVZoomedZoneTree::process_seq
 *****************************************************************************/

void PVParallelView::PVZoomedZoneTree::process_seq(const PVParallelView::PVZoneProcessing &zp)
{
	const uint32_t* pcol_a = zp.get_plotted_col_a();
	const uint32_t* pcol_b = zp.get_plotted_col_b();

	for (PVRow r = 0; r < zp.nrows(); ++r) {
		PVParallelView::PVQuadTreeEntry e(pcol_a[r], pcol_b[r], r);
		_trees[compute_index(e)].insert(e);
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedZoneTree::process_seq_from_zt
 *****************************************************************************/

void PVParallelView::PVZoomedZoneTree::process_seq_from_zt(const PVZoneProcessing &zp,
                                                           PVZoneTree &zt)
{
	register const uint32_t* pcol_a = zp.get_plotted_col_a();
	register const uint32_t* pcol_b = zp.get_plotted_col_b();

	for(unsigned i = 0; i < NBUCKETS; ++i) {
		pvquadtree& tree_i = _trees[i];
		for (unsigned j = 0; j < zt.get_branch_count(i); ++j) {
			const PVRow r = zt.get_branch_element(i, j);
			tree_i.insert(PVParallelView::PVQuadTreeEntry(pcol_a[r], pcol_b[r], r));
		}
		tree_i.compact();
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedZoneTree::process_omp
 *****************************************************************************/

void PVParallelView::PVZoomedZoneTree::process_omp(const PVParallelView::PVZoneProcessing &zp)
{
	const uint32_t* pcol_a = zp.get_plotted_col_a();
	const uint32_t* pcol_b = zp.get_plotted_col_b();
	const PVRow nrows = zp.nrows();

	uint32_t THREAD_ELE_COUNT = 64 / sizeof(PVQuadTreeEntry);
	const size_t nthreads = PVCore::PVHardwareConcurrency::get_physical_core_number();
	uint32_t STEP_ELE_COUNT = THREAD_ELE_COUNT * nthreads;
	uint32_t nrows_omp = (nrows / STEP_ELE_COUNT) * STEP_ELE_COUNT;
	uint32_t tree_count = (NBUCKETS) / nthreads;
	char *buffer = new char [nthreads * 64];

#pragma omp parallel num_threads(4)
	{
		unsigned tid = omp_get_thread_num();
		PVQuadTreeEntry *me = (PVQuadTreeEntry*) &(buffer)[tid * 64];
		PVQuadTreeEntry *e;
		unsigned tmin = tree_count * tid;
		unsigned tmax = tree_count * (tid + 1);

		for (PVRow r = 0; r < nrows_omp; r += STEP_ELE_COUNT) {
			unsigned base = r + tid * THREAD_ELE_COUNT;

			for (unsigned i = 0; i < THREAD_ELE_COUNT; ++i) {
				unsigned idx = base + i;
				me[i] = PVQuadTreeEntry(pcol_a[idx], pcol_b[idx], idx);
			}
#pragma omp barrier

			for (unsigned i = 0; i < nthreads; ++i) {
				e = (PVQuadTreeEntry*) &(buffer)[((tid+i) % nthreads) * 64];
				for (unsigned j = 0; j < THREAD_ELE_COUNT; ++j) {
					unsigned idx = compute_index(e[j]);
					if ((tmin < idx) && (idx < tmax)) {
						_trees[idx].insert(e[j]);
					}
				}
			}
#pragma omp barrier

		}
	}

	if (buffer) {
		delete buffer;
		buffer = 0;
	}

	// sequential end
	for (PVRow r = nrows_omp; r < nrows; ++r) {
		PVParallelView::PVQuadTreeEntry e(pcol_a[r], pcol_b[r], r);
		_trees[compute_index(e)].insert(e);
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedZoneTree::process_omp_from_zt
 *****************************************************************************/

void PVParallelView::PVZoomedZoneTree::process_omp_from_zt(const PVZoneProcessing &zp,
                                                           PVZoneTree &zt)
{
	const uint32_t* pcol_a = zp.get_plotted_col_a();
	const uint32_t* pcol_b = zp.get_plotted_col_b();

	BENCH_START(zztree);
#if 0
		for (size_t i = 0; i < NBUCKETS; i++) {
			for (size_t j = 0; j < treeb[i].count; ++j) {
				const PVRow r = treeb[i].p[j];

				PVParallelView::PVQuadTreeEntry e(pcol_a[r], pcol_b[r], r);
				this->_trees[i].insert(e);
			}
			this->_trees[i].compact();
		}
		BENCH_END(zztree, "ZZTREE CREATION (SERIAL)", 1, 1, 1, 1);
#else
		tbb::parallel_for(tbb::blocked_range<size_t>(0, NBUCKETS, 256), [&](tbb::blocked_range<size_t> const& range){
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
		}, tbb::auto_partitioner());
		BENCH_END(zztree, "ZZTREE CREATION (PARALLEL)", 1, 1, 1, 1);
#endif


}

/*****************************************************************************
 * PVParallelView::PVZoomedZoneTree::dump_to_file
 *****************************************************************************/

bool PVParallelView::PVZoomedZoneTree::dump_to_file(const char *filename) const
{
	FILE *fp = fopen(filename, "w");
	if (fp == NULL) {
		PVLOG_ERROR("Error while opening %s for writing: %s.\n",
		            filename, strerror(errno));
		return false;
	}

	for (size_t i = 0; i < NBUCKETS; ++i) {
		if (_trees[i].dump_to_file(filename, fp) == false) {
			return false;
		}
	}

	fclose(fp);
	return true;
}

/*****************************************************************************
 * PVParallelView::PVZoomedZoneTree::browse_trees_bci_by_y1
 *****************************************************************************/

PVParallelView::PVZoomedZoneTree *
PVParallelView::PVZoomedZoneTree::load_from_file(const char *filename)
{
	FILE *fp = fopen(filename, "r");
	if (fp == nullptr) {
		PVLOG_ERROR("Error while opening %s for reading: %s.\n",
		            filename, strerror(errno));
		return nullptr;
	}

	PVZoomedZoneTree *zzt = new PVZoomedZoneTree(nullptr, nullptr, 8);

	zzt->_trees = new pvquadtree [NBUCKETS];

	bool err = false;
	for(size_t i = 0; i < NBUCKETS; ++i) {
		if (zzt->_trees[i].load_from_file(filename, fp) == false) {
			err = true;
			break;
		}
	}

	fclose(fp);

	if (err) {
		delete zzt;
		zzt = nullptr;
	}

	return zzt;
}

/*****************************************************************************
 * PVParallelView::PVZoomedZoneTree::browse_trees_bci_by_y1
 *****************************************************************************/

size_t PVParallelView::PVZoomedZoneTree::browse_trees_bci_by_y1_seq(context_t &ctx,
                                                                    uint64_t y_min,
                                                                    uint64_t y_max,
                                                                    uint64_t y_lim,
                                                                    int zoom,
                                                                    uint32_t width,
                                                                    const extract_entries_f &extract_f,
                                                                    const PVCore::PVHSVColor *colors,
                                                                    pv_bci_code_t *codes,
                                                                    const float beta,
                                                                    const PVRow* sel_elts) const
{
	uint32_t shift = (32 - bbits) - zoom;
	uint32_t t1_min = y_min >> (32 - NBITS_INDEX);
	uint32_t t1_max = (uint32_t)PVCore::clamp<uint64_t>(1 + (y_max >> (32 - NBITS_INDEX)),
	                                                    0U, 1024U);
	zzt_tls &tls = ctx.get_tls().local();
	pv_tlr_buffer_t &tlr_buffer = tls.get_tlr_buffer();
	pv_quadtree_buffer_entry_t *quadtree_buffer = tls.get_quadtree_buffer();

	const insert_entry_f insert_f =
		insert_entry_f([&](const PVQuadTreeEntry &e, pv_tlr_buffer_t &buffer)
		               {
			               pv_bci_code_t bci;
			               compute_bci_projection_y2(e.y1, e.y2,
			                                         y_min, y_lim,
			                                         shift, mask_int_ycoord,
			                                         width, beta, bci);
			               pv_tlr_index_t tlr(bci.s.type,
			                                  bci.s.l,
			                                  bci.s.r);
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
				 * drawn selected line for the corresponding ZoneTree, it is useless
				 * to search for a selected line in the quadtree
				 */
				continue;
			}

			/* compute the events number along the secondary coordinate
			 */
			const uint32_t y2_count =
				compute_sec_coord_count_y2(t1, t2,
				                           y_min, y_lim,
				                           shift, mask_int_ycoord,
				                           zoom, width, beta);

			/* lines extraction
			 */
			extract_f(_trees[tree_idx], y2_count,
			          quadtree_buffer, tlr_buffer, insert_f);
		}
	}
	BENCH_STOP(extract);

	/* extracting BCI codes from TLR buffer
	 */
	size_t bci_idx = 0;

	BENCH_START(compute);
	for(size_t i = 0; i < pv_tlr_buffer_t::length; ++i) {
		const pv_tlr_index_t tlr(i);

		const uint32_t idx = tlr_buffer[i];
		if (idx != UINT32_MAX) {
			pv_bci_code_t &bci = codes[bci_idx];
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
		BENCH_SHOW(whole,   "whole       y1 seq sel", 1, 1, 1, 1);
	} else {
		BENCH_SHOW(extract, "extraction  y1 seq bg", 1, 1, 1, 1);
		BENCH_SHOW(compute, "computation y1 seq bg", 1, 1, 1, 1);
		BENCH_SHOW(whole,   "whole       y1 seq bg", 1, 1, 1, 1);
	}

	PVLOG_DEBUG("::browse_trees_bci_by_y1_seq -> %lu\n", bci_idx);

	return bci_idx;
}

/*****************************************************************************
 * PVParallelView::PVZoomedZoneTree::browse_trees_bci_by_y2
 *****************************************************************************/

size_t PVParallelView::PVZoomedZoneTree::browse_trees_bci_by_y2_seq(context_t &ctx,
                                                                    uint64_t y_min,
                                                                    uint64_t y_max,
                                                                    uint64_t y_lim,
                                                                    int zoom,
                                                                    uint32_t width,
                                                                    const extract_entries_f &extract_f,
                                                                    const PVCore::PVHSVColor *colors,
                                                                    pv_bci_code_t *codes,
                                                                    const float beta,
                                                                    const PVRow* sel_elts) const
{
	uint32_t shift = (32 - bbits) - zoom;
	uint32_t t2_min = y_min >> (32 - NBITS_INDEX);
	uint32_t t2_max = (uint32_t)PVCore::clamp<uint64_t>(1 + (y_max >> (32 - NBITS_INDEX)),
	                                                    0U, 1024U);
	zzt_tls &tls = ctx.get_tls().local();
	pv_tlr_buffer_t &tlr_buffer = tls.get_tlr_buffer();
	pv_quadtree_buffer_entry_t *quadtree_buffer = tls.get_quadtree_buffer();

	const insert_entry_f insert_f =
		insert_entry_f([&](const PVQuadTreeEntry &e, pv_tlr_buffer_t &buffer)
		               {
			               pv_bci_code_t bci;
			               compute_bci_projection_y1(e.y1, e.y2,
			                                         y_min, y_lim,
			                                         shift, mask_int_ycoord,
			                                         width, beta, bci);
			               pv_tlr_index_t tlr(bci.s.type,
			                                  bci.s.l,
			                                  bci.s.r);
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
				 * drawn selected line for the corresponding ZoneTree, it is useless
				 * to search for a selected line in the quadtree
				 */
				continue;
			}

			/* compute the events number along the secondary coordinate
			 */
			const uint32_t y1_count =
				compute_sec_coord_count_y1(t1, t2,
				                           y_min, y_lim,
				                           shift, mask_int_ycoord,
				                           zoom, width, beta);

			/* lines extraction
			 */
			extract_f(_trees[tree_idx], y1_count,
			          quadtree_buffer, tlr_buffer, insert_f);
		}
	}
	BENCH_STOP(extract);

	/* extracting BCI codes from TLR buffer
	 */
	size_t bci_idx = 0;

	BENCH_START(compute);
	for(size_t i = 0; i < pv_tlr_buffer_t::length; ++i) {
		const pv_tlr_index_t tlr(i);

		const uint32_t idx = tlr_buffer[i];
		if (idx != UINT32_MAX) {
			pv_bci_code_t &bci = codes[bci_idx];
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
		BENCH_SHOW(whole,   "whole       y2 seq sel", 1, 1, 1, 1);
	} else {
		BENCH_SHOW(extract, "extraction  y2 seq bg", 1, 1, 1, 1);
		BENCH_SHOW(compute, "computation y2 seq bg", 1, 1, 1, 1);
		BENCH_SHOW(whole,   "whole       y2 seq sel", 1, 1, 1, 1);
	}

	PVLOG_DEBUG("::browse_trees_bci_by_y2_seq -> %lu\n", bci_idx);

	return bci_idx;
}

/*****************************************************************************
 * PVParallelView::PVZoomedZoneTree::browse_trees_bci_by_y1_tbb
 *****************************************************************************/

size_t PVParallelView::PVZoomedZoneTree::browse_trees_bci_by_y1_tbb(context_t &ctx,
                                                                    uint64_t y_min,
                                                                    uint64_t y_max,
                                                                    uint64_t y_lim,
                                                                    int zoom,
                                                                    uint32_t width,
                                                                    const extract_entries_f &extract_f,
                                                                    const PVCore::PVHSVColor *colors,
                                                                    pv_bci_code_t *codes,
                                                                    const float beta,
                                                                    const PVRow* sel_elts) const
{
	uint32_t shift = (32 - bbits) - zoom;
	uint32_t t1_min = y_min >> (32 - NBITS_INDEX);
	uint32_t t1_max = (uint32_t)PVCore::clamp<uint64_t>(1 + (y_max >> (32 - NBITS_INDEX)),
	                                                    0U, 1024U);

	const insert_entry_f insert_f =
		insert_entry_f([&](const PVQuadTreeEntry &e, pv_tlr_buffer_t &buffer)
		               {
			               pv_bci_code_t bci;
			               compute_bci_projection_y2(e.y1, e.y2,
			                                         y_min, y_lim,
			                                         shift, mask_int_ycoord,
			                                         width, beta, bci);
			               pv_tlr_index_t tlr(bci.s.type,
			                                  bci.s.l,
			                                  bci.s.r);
			               if (e.idx < buffer[tlr.v]) {
				               buffer[tlr.v] = e.idx;
			               }
		               });

	BENCH_START(whole);
	BENCH_START(extract);
	tbb::parallel_for(tbb::blocked_range2d<uint32_t>(t1_min, t1_max, 0, 1024),
	                  [&] (const tbb::blocked_range2d<uint32_t> &r)
	                  {
		                  zzt_tls &tls = ctx.get_tls().local();
		                  pv_tlr_buffer_t &tlr_buffer = tls.get_tlr_buffer();
		                  pv_quadtree_buffer_entry_t *quadtree_buffer = tls.get_quadtree_buffer();

						  // AG: copy this variable to the local stack (or better, register), which may reduce the number
						  // of loads from the original stack.
						  const PVRow* sel_elts_ = sel_elts;
		                  for (uint32_t t1 = r.rows().begin(); t1 != r.rows().end(); ++t1) {
			                  for (uint32_t t2 = r.cols().begin(); t2 != r.cols().end(); ++t2) {
				                  PVRow tree_idx = (t2 * 1024) + t1;

				                  if (sel_elts_ && (sel_elts_[tree_idx] == PVROW_INVALID_VALUE)) {
					                  /* when searching for entries using the selection, if there is no
					                   * drawn selected line for the corresponding ZoneTree, it is useless
					                   * to search for a selected line in the quadtree
					                   */
					                  continue;
				                  }

				                  /* compute the events number along the secondary coordinate
				                   */
				                  const uint32_t y2_count =
					                  compute_sec_coord_count_y2(t1, t2,
					                                             y_min, y_lim,
					                                             shift, mask_int_ycoord,
					                                             zoom, width, beta);

				                  /* lines extraction
				                   */
				                  extract_f(_trees[tree_idx], y2_count,
				                            quadtree_buffer, tlr_buffer, insert_f);
			                  }
		                  }
	                  });
	BENCH_STOP(extract);

	/* merging all TLR buffers
	 */
	BENCH_START(merge);
	context_t::tls_set_t::iterator it = ctx.get_tls().begin();
	pv_tlr_buffer_t &tlr_buffer = it->get_tlr_buffer();

	++it;
	for(;it != ctx.get_tls().end(); ++it) {
		pv_tlr_buffer_t &tlr_buffer2 = it->get_tlr_buffer();

		for(size_t i = 0; i < pv_tlr_buffer_t::length; ++i) {
			if(tlr_buffer2[i] < tlr_buffer[i]) {
				tlr_buffer[i] = tlr_buffer2[i];
			}
		}

		tlr_buffer2.clear();
	}
	BENCH_STOP(merge);

	/* extracting BCI codes from TLR buffer
	 */
	size_t bci_idx = 0;

	BENCH_START(compute);
	for(size_t i = 0; i < pv_tlr_buffer_t::length; ++i) {
		const pv_tlr_index_t tlr(i);

		const uint32_t idx = tlr_buffer[i];
		if (idx != UINT32_MAX) {
			pv_bci_code_t &bci = codes[bci_idx];
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
		BENCH_SHOW(merge,   "merge       y1 tbb sel", 1, 1, 1, 1);
		BENCH_SHOW(compute, "computation y1 tbb sel", 1, 1, 1, 1);
		BENCH_SHOW(whole,   "whole       y1 tbb sel", 1, 1, 1, 1);
	} else {
		BENCH_SHOW(extract, "extraction  y1 tbb bg", 1, 1, 1, 1);
		BENCH_SHOW(merge,   "merge       y1 tbb bg", 1, 1, 1, 1);
		BENCH_SHOW(compute, "computation y1 tbb bg", 1, 1, 1, 1);
		BENCH_SHOW(whole,   "whole       y1 tbb bg", 1, 1, 1, 1);
	}

	PVLOG_DEBUG("::browse_trees_bci_by_y1_tbb -> %lu\n", bci_idx);

	return bci_idx;
}

/*****************************************************************************
 * PVParallelView::PVZoomedZoneTree::browse_trees_bci_by_y2_tbb
 *****************************************************************************/

size_t PVParallelView::PVZoomedZoneTree::browse_trees_bci_by_y2_tbb(context_t &ctx,
                                                                    uint64_t y_min,
                                                                    uint64_t y_max,
                                                                    uint64_t y_lim,
                                                                    int zoom,
                                                                    uint32_t width,
                                                                    const extract_entries_f &extract_f,
                                                                    const PVCore::PVHSVColor *colors,
                                                                    pv_bci_code_t *codes,
                                                                    const float beta,
                                                                    const PVRow* sel_elts) const
{
	uint32_t shift = (32 - bbits) - zoom;
	uint32_t t2_min = y_min >> (32 - NBITS_INDEX);
	uint32_t t2_max = (uint32_t)PVCore::clamp<uint64_t>(1 + (y_max >> (32 - NBITS_INDEX)),
	                                                    0U, 1024U);


	const insert_entry_f insert_f =
		insert_entry_f([&](const PVQuadTreeEntry &e, pv_tlr_buffer_t &buffer)
		               {
			               pv_bci_code_t bci;
			               compute_bci_projection_y1(e.y1, e.y2,
			                                         y_min, y_lim,
			                                         shift, mask_int_ycoord,
			                                         width, beta, bci);
			               pv_tlr_index_t tlr(bci.s.type,
			                                  bci.s.l,
			                                  bci.s.r);
			               if (e.idx < buffer[tlr.v]) {
				               buffer[tlr.v] = e.idx;
			               }
		               });

	BENCH_START(whole);
	BENCH_START(extract);
	tbb::parallel_for(tbb::blocked_range2d<uint32_t>(t2_min, t2_max, 0, 1024),
	                  [&] (const tbb::blocked_range2d<uint32_t> &r)
	                  {
		                  zzt_tls &tls = ctx.get_tls().local();
		                  pv_tlr_buffer_t &tlr_buffer = tls.get_tlr_buffer();
		                  pv_quadtree_buffer_entry_t *quadtree_buffer = tls.get_quadtree_buffer();

						  // AG: this copy is wanted. cf. browse_trees_bci_by_y1_tbb.
						  const PVRow* sel_elts_ = sel_elts;
		                  for (uint32_t t2 = r.rows().begin(); t2 != r.rows().end(); ++t2) {
			                  for (uint32_t t1 = r.cols().begin(); t1 != r.cols().end(); ++t1) {
				                  PVRow tree_idx = (t2 * 1024) + t1;

				                  if (sel_elts_ && (sel_elts_[tree_idx] == PVROW_INVALID_VALUE)) {
					                  /* when searching for entries using the selection, if there is no
					                   * drawn selected line for the corresponding ZoneTree, it is useless
					                   * to search for a selected line in the quadtree
					                   */
					                  continue;
				                  }

				                  /* compute the events number along the secondary coordinate
				                   */
				                  const uint32_t y1_count =
					                  compute_sec_coord_count_y1(t1, t2,
					                                             y_min, y_lim,
					                                             shift, mask_int_ycoord,
					                                             zoom, width, beta);

				                  /* lines extraction
				                   */
				                  extract_f(_trees[tree_idx], y1_count,
				                            quadtree_buffer, tlr_buffer, insert_f);
			                  }
		                  }
	                  });
	BENCH_STOP(extract);

	/* merging all TLR buffers
	 */
	BENCH_START(merge);
	context_t::tls_set_t::iterator it = ctx.get_tls().begin();
	pv_tlr_buffer_t &tlr_buffer = it->get_tlr_buffer();

	++it;
	for(;it != ctx.get_tls().end(); ++it) {
		pv_tlr_buffer_t &tlr_buffer2 = it->get_tlr_buffer();

		for(size_t i = 0; i < pv_tlr_buffer_t::length; ++i) {
			if(tlr_buffer2[i] < tlr_buffer[i]) {
				tlr_buffer[i] = tlr_buffer2[i];
			}
		}

		tlr_buffer2.clear();
	}
	BENCH_STOP(merge);

	/* extracting BCI codes from TLR buffer
	 */
	size_t bci_idx = 0;

	BENCH_START(compute);
	for(size_t i = 0; i < pv_tlr_buffer_t::length; ++i) {
		const pv_tlr_index_t tlr(i);

		const uint32_t idx = tlr_buffer[i];
		if (idx != UINT32_MAX) {
			pv_bci_code_t &bci = codes[bci_idx];
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
		BENCH_SHOW(merge,   "merge       y2 tbb sel", 1, 1, 1, 1);
		BENCH_SHOW(compute, "computation y2 tbb sel", 1, 1, 1, 1);
		BENCH_SHOW(whole,   "whole       y2 tbb sel", 1, 1, 1, 1);
	} else {
		BENCH_SHOW(extract, "extraction  y2 tbb bg", 1, 1, 1, 1);
		BENCH_SHOW(merge,   "merge       y2 tbb bg", 1, 1, 1, 1);
		BENCH_SHOW(compute, "computation y2 tbb bg", 1, 1, 1, 1);
		BENCH_SHOW(whole,   "whole       y2 tbb bg", 1, 1, 1, 1);
	}

	PVLOG_DEBUG("::browse_trees_bci_by_y2_tbb -> %lu\n", bci_idx);

	return bci_idx;
}

#ifdef PICVIZ_DEVELOPER_MODE
double PVParallelView::extract_stat::all_dt = 0;
size_t PVParallelView::extract_stat::all_cnt = 0;
size_t PVParallelView::extract_stat::test_cnt = 0;
size_t PVParallelView::extract_stat::insert_cnt = 0;
#endif
