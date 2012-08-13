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
	PVParallelView::PVZoneTree::PVBranch *treeb = zt.get_treeb();
	PVRow r;

	for(unsigned i = 0; i < NBUCKETS; ++i) {
		for (unsigned j = 0; j < treeb[i].count; ++j) {
			r = treeb[i].p[j];
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
	uint32_t nthreads = 4;
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
	PVParallelView::PVZoneTree::PVBranch *treeb = zt.get_treeb();

	const size_t nthreads = PVCore::PVHardwareConcurrency::get_physical_core_number();

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
				for (size_t j = 0; j < treeb[i].count; ++j) {
					const PVRow r = treeb[i].p[j];

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

size_t PVParallelView::PVZoomedZoneTree::browse_tree_bci_by_y1(uint32_t y_min, int zoom,
                                                               const PVHSVColor* colors,
                                                               PVBCICode<bbits>* codes,
                                                               const float beta) const
{
	uint32_t t_min = y_min >> (32 - NBITS_INDEX);
	uint32_t t_max = PVCore::clamp<uint32_t>(t_min + (1024U >> zoom), 0U, 1024U);
	uint32_t y_max = (uint32_t) PVCore::clamp<uint64_t>(y_min + ((uint64_t)UINT32_MAX >> zoom),
	                                                    0, UINT32_MAX);
	size_t num = 0;

	for (uint32_t t1 = t_min; t1 < t_max; ++t1) {
		for (uint32_t t2 = 0; t2 < 1024; ++t2) {
			num += _trees[(t2 * 1024) + t1].get_first_bci_from_y1(y_min, y_max, zoom, colors, codes + num);
		}
	}

#pragma omp parallel for num_threads(4) // usefull?
	for(size_t i = 0; i < num; ++i) {
		int l = (int)codes[i].s.l;
		int r = (int)codes[i].s.r;
		int d = (r - l) * beta;
		codes[i].s.r = l + d;
	}

	std::cout << "::browse_tree_bci_by_y1 -> " << num << std::endl;

	return num;
}

/*****************************************************************************
 * PVParallelView::PVZoomedZoneTree::browse_tree_bci_by_y2
 *****************************************************************************/

size_t PVParallelView::PVZoomedZoneTree::browse_tree_bci_by_y2(uint32_t y_min, int zoom,
                                                               const PVHSVColor* colors,
                                                               PVBCICode<bbits>* codes,
                                                               const float beta) const
{
	uint32_t t_min = y_min >> (32 - NBITS_INDEX);
	uint32_t t_max = PVCore::clamp<uint32_t>(t_min + (1024U >> zoom), 0U, 1024U);
	uint32_t y_max = (uint32_t) PVCore::clamp<uint64_t>(y_min + ((uint64_t)UINT32_MAX >> zoom),
	                                                    0, UINT32_MAX);
	size_t num = 0;

	for (uint32_t t1 = 0; t1 < 1024; ++t1) {
		for (uint32_t t2 = t_min; t2 < t_max; ++t2) {
			num += _trees[(t2 * 1024) + t1].get_first_bci_from_y2(y_min, y_max, zoom, colors, codes + num);
		}
	}

#pragma omp parallel for num_threads(4) // usefull?
	for(size_t i = 0; i < num; ++i) {
		int l = (int)codes[i].s.l;
		int r = (int)codes[i].s.r;
		int d = (l - r) * beta;
		codes[i].s.l = r + d;
	}

	std::cout << "::browse_tree_bci_by_y2 -> " << num << std::endl;

	return num;
}

/*****************************************************************************
 * PVParallelView::PVZoomedZoneTree::browse_tree_bci_by_y1_range
 *****************************************************************************/

size_t PVParallelView::PVZoomedZoneTree::browse_tree_bci_by_y1_range(uint32_t y_min,
                                                                     uint32_t y_max,
                                                                     int zoom,
                                                                     const PVHSVColor* colors,
                                                                     PVBCICode<bbits>* codes,
                                                                     const float beta) const
{
	uint32_t t1_min = y_min >> (32 - NBITS_INDEX);
	uint32_t t1_max = PVCore::clamp<uint32_t>(t1_min + (1024U >> zoom), 0U, 1024U);
	size_t num = 0, old_num;
	uint32_t shift = (32 - bbits) - zoom;

	for (uint32_t t1 = t1_min; t1 < t1_max; ++t1) {
		uint32_t t2_min = (uint32_t)PVCore::clamp<double>(t1 - ((t1 - t1_min) / (double)beta),
		                                                  0., 1024.);
		uint32_t t2_max = (uint32_t)PVCore::clamp<double>(t1 + ((t1_max - t1) / (double)beta),
		                                                  0., 1024.);
		// std::cout << "t1 -> t2_m{in,ax}: "
		//           << t1 << " "
		//           << t2_min << " " << t2_max << std::endl;

		for (uint32_t t2 = t2_min; t2 < t2_max; ++t2) {
			old_num = num;
			// TODO: "translate" BCI codes
			num += _trees[(t2 * 1024) + t1].get_first_from_y1(y_min, y_max, zoom,
			                                                  colors, _quad_entries + old_num);

			for(size_t i = old_num; i < num; ++i) {
				PVParallelView::PVQuadTreeEntry &e = _quad_entries[i];
				codes[i].s.l = ((e.y1 - y_min) >> shift) & mask_int_ycoord;
				codes[i].s.color = colors[e.idx].h();

				int64_t r = e.y1 + (e.y2 - e.y1) * beta;
				codes[i].s.r = ((r - y_min) >> shift) & mask_int_ycoord;
			}
		}
	}

	std::cout << "::browse_tree_bci_by_y1_range -> " << num << std::endl;

	return num;
}
