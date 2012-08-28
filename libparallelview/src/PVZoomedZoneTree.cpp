/**
 * \file PVZoomedZoneTree.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <iostream>

#include <omp.h>

#include <pvkernel/core/PVAlgorithms.h>
#include <pvparallelview/PVZoomedZoneTree.h>

#include <tbb/parallel_for.h>
#include <tbb/task_scheduler_init.h>

#define ZZT_MAX_VALUE (1 << (32-NBITS_INDEX))

/*****************************************************************************
 * PVParallelView::PVZoomedZoneTree::PVZoomedZoneTree
 *****************************************************************************/

PVParallelView::PVZoomedZoneTree::PVZoomedZoneTree(uint32_t max_level)
{
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
			                 max_level);
			y1_min += ZZT_MAX_VALUE;
			++idx;
		}
		y2_min += ZZT_MAX_VALUE;
	}

	// TODO: find an appropriate size
	_quad_entries = new PVParallelView::PVQuadTreeEntry [NBUCKETS];
}

/*****************************************************************************
 * PVParallelView::PVZoomedZoneTree::~PVZoomedZoneTree
 *****************************************************************************/

PVParallelView::PVZoomedZoneTree::~PVZoomedZoneTree()
{
	if (_trees != 0) {
		delete [] _trees;
	}
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
	const uint32_t* pcol_a = zp.get_plotted_col_a();
	const uint32_t* pcol_b = zp.get_plotted_col_b();
	PVRow r;

	for(unsigned i = 0; i < NBUCKETS; ++i) {
		for (unsigned j = 0; j < zt.get_branch_count(i); ++j) {
			r = zt.get_branch_element(i, j);
			PVParallelView::PVQuadTreeEntry e(pcol_a[r], pcol_b[r], r);
			_trees[i].insert(e);
		}
		_trees[i].compact();
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
		tbb::parallel_for(tbb::blocked_range<size_t>(0, NBUCKETS, 128), [&](tbb::blocked_range<size_t> const& range){
			for (size_t i = range.begin(); i != range.end(); i++) {
				for (size_t j = 0; j < zt.get_branch_count(i); ++j) {
					const PVRow r = zt.get_branch_element(i, j);

					PVParallelView::PVQuadTreeEntry e(pcol_a[r], pcol_b[r], r);
					this->_trees[i].insert(e);
				}
				this->_trees[i].compact();
			}
		});
		BENCH_END(zztree, "ZZTREE CREATION (PARALLEL)", 1, 1, 1, 1);
#endif


}

/*****************************************************************************
 * PVParallelView::PVZoomedZoneTree::browse_tree_bci_by_y1
 *****************************************************************************/

size_t PVParallelView::PVZoomedZoneTree::browse_tree_bci_by_y1(uint64_t y_min,
                                                               uint64_t y_max,
                                                               uint64_t y_lim,
                                                               int zoom,
                                                               uint32_t width,
                                                               const PVCore::PVHSVColor* colors,
                                                               PVBCICode<bbits>* codes,
                                                               const float beta) const
{
	size_t num = 0;
	size_t bci_idx = 0;
	uint32_t shift = (32 - bbits) - zoom;
	uint32_t t1_min = y_min >> (32 - NBITS_INDEX);
	uint32_t t1_max = (uint32_t)PVCore::clamp<uint64_t>(1 + (y_max >> (32 - NBITS_INDEX)),
	                                                    0U, 1024U);

	for (uint32_t t1 = t1_min; t1 < t1_max; ++t1) {
		for (uint32_t t2 = 0; t2 < 1024; ++t2) {
			/* lines extraction
			 */
			num = _trees[(t2 * 1024) + t1].get_first_from_y1(y_min, y_max, zoom,
			                                                 colors, _quad_entries);

			/* conversion into BCI codes
			 */
			for (size_t e_idx = 0; e_idx < num; ++e_idx) {
				PVBCICode<bbits> bci;
				PVParallelView::PVQuadTreeEntry &e = _quad_entries[e_idx];

				bci.s.idx = e.idx;
				bci.s.color = colors[e.idx].h();
				bci.s.l = ((e.y1 - y_min) >> shift) & mask_int_ycoord;

				int64_t d = (int64_t)e.y2 - (int64_t)e.y1;
				uint32_t y2p = (uint32_t)((int64_t)e.y1 + d * beta);

				if (y2p >= y_lim) {
					bci.s.type = PVParallelView::PVBCICode<bbits>::DOWN;
					bci.s.r = (double)width * ((double)(y_lim - e.y1) / (double)(y2p - e.y1));
				} else if (y2p <= y_min) {
					bci.s.type = PVParallelView::PVBCICode<bbits>::UP;
					bci.s.r = (double)width * ((double)(e.y1 - y_min) / (double)(e.y1 - y2p));
				} else {
					bci.s.type = PVParallelView::PVBCICode<bbits>::STRAIGHT;
					bci.s.r = ((y2p - y_min) >> shift) & mask_int_ycoord;
				}

				/* zoom make some entries having the same BCI codes. It is also useless
				 * to render all of them.
				 */
				if (bci_idx == 0) {
					// first entry, insert it!
					codes[bci_idx] = bci;
					++bci_idx;
				} else if ((codes[bci_idx-1].s.l != bci.s.l)
				           ||
				           (codes[bci_idx-1].s.r != bci.s.r)
				           ||
				           (codes[bci_idx-1].s.type != bci.s.type)) {
					// the BCI code is a new one, insert it!
					codes[bci_idx] = bci;
					++bci_idx;
				} else {
					// same BCI code
					if (bci.s.idx < codes[bci_idx-1].s.idx) {
						// we want the entry with the lowest index
						codes[bci_idx-1] = bci;
					}
				}

			}
		}
	}

	std::cout << "::browse_tree_bci_by_y1 -> " << bci_idx << std::endl;

	return bci_idx;
}

/*****************************************************************************
 * PVParallelView::PVZoomedZoneTree::browse_tree_bci_by_y2
 *****************************************************************************/

size_t PVParallelView::PVZoomedZoneTree::browse_tree_bci_by_y2(uint64_t y_min,
                                                               uint64_t y_max,
                                                               uint64_t y_lim,
                                                               int zoom,
                                                               uint32_t width,
                                                               const PVCore::PVHSVColor* colors,
                                                               PVBCICode<bbits>* codes,
                                                               const float beta) const
{
	size_t num = 0;
	size_t bci_idx = 0;
	uint32_t shift = (32 - bbits) - zoom;
	uint32_t t2_min = y_min >> (32 - NBITS_INDEX);
	uint32_t t2_max = (uint32_t)PVCore::clamp<uint64_t>(1 + (y_max >> (32 - NBITS_INDEX)),
	                                                    0U, 1024U);

	for (uint32_t t2 = t2_min; t2 < t2_max; ++t2) {
		for (uint32_t t1 = 0; t1 < 1024; ++t1) {
			/* lines extraction
			 */
			num = _trees[(t2 * 1024) + t1].get_first_from_y2(y_min, y_max, zoom,
			                                                 colors, _quad_entries);

			/* conversion into BCI codes
			 */
			for (size_t e_idx = 0; e_idx < num; ++e_idx) {
				PVBCICode<bbits> bci;
				PVParallelView::PVQuadTreeEntry &e = _quad_entries[e_idx];

				bci.s.idx = e.idx;
				bci.s.color = colors[e.idx].h();
				bci.s.l = ((e.y2 - y_min) >> shift) & mask_int_ycoord;

				int64_t d = (int64_t)e.y1 - (int64_t)e.y2;
				uint32_t y1p = (uint32_t)((int64_t)e.y2 + d * beta);

				if (y1p >= y_lim) {
					bci.s.type = PVParallelView::PVBCICode<bbits>::DOWN;
					bci.s.r = (double)width * ((double)(y_lim - e.y2) / (double)(y1p - e.y2));
				} else if (y1p <= y_min) {
					bci.s.type = PVParallelView::PVBCICode<bbits>::UP;
					bci.s.r = (double)width * ((double)(e.y2 - y_min) / (double)(e.y2 - y1p));
				} else {
					bci.s.type = PVParallelView::PVBCICode<bbits>::STRAIGHT;
					bci.s.r = ((y1p - y_min) >> shift) & mask_int_ycoord;
				}

				/* zoom make some entries having the same BCI codes. It is also useless
				 * to render all of them.
				 */
				if (bci_idx == 0) {
					// first entry, insert it!
					codes[bci_idx] = bci;
					++bci_idx;
				} else if ((codes[bci_idx-1].s.l != bci.s.l)
				           ||
				           (codes[bci_idx-1].s.r != bci.s.r)
				           ||
				           (codes[bci_idx-1].s.type != bci.s.type)) {
					// the BCI code is a new one, insert it!
					codes[bci_idx] = bci;
					++bci_idx;
				} else {
					// same BCI code
					if (bci.s.idx < codes[bci_idx-1].s.idx) {
						// we want the entry with the lowest index
						codes[bci_idx-1] = bci;
					}
				}

			}
		}
	}

	std::cout << "::browse_tree_bci_by_y2 -> " << bci_idx << std::endl;

	return bci_idx;
}
