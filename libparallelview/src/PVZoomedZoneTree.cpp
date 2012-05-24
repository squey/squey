
#include <iostream>

#include <omp.h>

#include <pvparallelview/PVZoomedZoneTree.h>

#define ZZT_MAX_VALUE (1 << (32-NBITS_INDEX))

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
}

PVParallelView::PVZoomedZoneTree::~PVZoomedZoneTree()
{
	if (_trees != 0) {
		delete [] _trees;
	}
}

void PVParallelView::PVZoomedZoneTree::process_seq(const PVParallelView::PVZoneProcessing &zp)
{
	const uint32_t* pcol_a = zp.get_plotted_col_a();
	const uint32_t* pcol_b = zp.get_plotted_col_b();

	for (PVRow r = 0; r < zp.nrows(); ++r) {
		PVParallelView::PVQuadTreeEntry e(pcol_a[r], pcol_b[r], r);
		_trees[compute_index(e)].insert(e);
	}
}

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

void PVParallelView::PVZoomedZoneTree::process_omp(const PVParallelView::PVZoneProcessing &zp)
{
	const uint32_t* pcol_a = zp.get_plotted_col_a();
	const uint32_t* pcol_b = zp.get_plotted_col_b();
	const PVRow nrows = zp.nrows();

	uint32_t THREAD_ELE_COUNT = 64 / sizeof(PVQuadTreeEntry);
	uint32_t nthreads = 4;
	uint32_t STEP_ELE_COUNT = THREAD_ELE_COUNT * nthreads;
	uint32_t nrows_omp = (nrows / STEP_ELE_COUNT) * STEP_ELE_COUNT;
	uint32_t tree_count = (1024 * 1024) / nthreads;
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

void PVParallelView::PVZoomedZoneTree::process_omp_from_zt(const PVZoneProcessing &zp,
                                                           PVZoneTree &zt)
{
	const uint32_t* pcol_a = zp.get_plotted_col_a();
	const uint32_t* pcol_b = zp.get_plotted_col_b();
	PVParallelView::PVZoneTree::PVBranch *treeb = zt.get_treeb();
	PVRow r;

#pragma omp parallel for num_threads(4)
	for(unsigned i = 0; i < NBUCKETS; ++i) {
		for (unsigned j = 0; j < treeb[i].count; ++j) {
			r = treeb[i].p[j];
			PVParallelView::PVQuadTreeEntry e(pcol_a[r], pcol_b[r], r);
			_trees[i].insert(e);
		}
		_trees[i].compact();
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
