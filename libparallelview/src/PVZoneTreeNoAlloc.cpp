/**
 * \file PVZoneTreeNoAlloc.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVHardwareConcurrency.h>
#include <pvkernel/core/picviz_bench.h>
#include <pvkernel/core/picviz_intrin.h>

#include <pvparallelview/PVBCode.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVHSVColor.h>
#include <pvparallelview/PVZoneTreeNoAlloc.h>
#include <pvparallelview/PVZoneProcessing.h>
#include <pvparallelview/simple_lines_float_view.h>

#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_for.h>

#include <cassert>

#include <omp.h>

#define GRAINSIZE 100

namespace __impl {

class TBBPF {
public:
	TBBPF (
		PVParallelView::PVZoneTreeNoAlloc* tree,
		const Picviz::PVSelection::const_pointer sel_buf,
		PVParallelView::PVZoneTreeNoAlloc::TLS* tls
	) :
		_tree(tree),
		_sel_buf(sel_buf),
		_tls(tls)
	{
	}

	TBBPF (TBBPF& x, tbb::split) :  _tree(x._tree), _sel_buf(x._sel_buf), _tls(x._tls)
	{}

	void operator() (const tbb::blocked_range<size_t>& r) const {
		PVParallelView::PVZoneTreeNoAlloc::TLS::reference tls_ref = _tls->local();
		tls_ref.reserve(r.size());
		for (PVRow b = r.begin(); b != r.end(); ++b) {
			if (_tree->branch_valid(b)) {
				PVRow r = _tree->get_first_elt_of_branch(b);
				bool found = false;
				if ((_sel_buf[r>>5]) & (1U<<(r&31))) {
					found = true;
				}
				else {
					PVParallelView::PVZoneTreeNoAlloc::Tree::const_branch_iterator it_src = _tree->_tree.begin_branch(b);
					it_src++;
					for (; it_src != _tree->_tree.end_branch(b); it_src++) {
						r = *(it_src);
						if ((_sel_buf[r>>5]) & (1U<<(r&31))) {
							found = true;
							break;
						}
					}
				}
				if (found) {
					_tree->_sel_elts[b] = r;
				}
				else {
					_tree->_sel_elts[b] = PVROW_INVALID_VALUE;
				}
			}
		}
	}

	mutable PVParallelView::PVZoneTreeNoAlloc* _tree;
	Picviz::PVSelection::const_pointer _sel_buf;
	PVParallelView::PVZoneTreeNoAlloc::TLS* _tls;
};

}

size_t PVParallelView::PVZoneTreeNoAlloc::browse_tree_bci_by_sel(PVHSVColor* colors, PVBCICode<NBITS_INDEX>* codes, Picviz::PVSelection const& sel)
{
	size_t idx_code = 0;
	Picviz::PVSelection::const_pointer sel_buf = sel.get_buffer();
	const size_t nthreads = PVCore::PVHardwareConcurrency::get_physical_core_number();
#pragma omp parallel firstprivate(sel_buf) reduction(+:idx_code) num_threads(nthreads)
	{
		PVBCICode<NBITS_INDEX>* th_codes = PVBCICode<NBITS_INDEX>::allocate_codes(NBUCKETS);
#pragma omp for schedule(dynamic, 6)
		for (uint64_t b = 0; b < NBUCKETS; b++) {
			if (branch_valid(b)) {
				PVRow r = get_first_elt_of_branch(b);
				bool found = false;
				if ((sel_buf[r>>5]) & (1U<<(r&31))) {
					found = true;
				}
				else {
					Tree::const_branch_iterator it_src = _tree.begin_branch(b);
					it_src++;
					for (; it_src != _tree.end_branch(b); it_src++) {
						r = *(it_src);
						if ((sel_buf[r>>5]) & (1U<<(r&31))) {
							found = true;
							break;
						}
					}
				}
				if (found) {
					PVBCICode<NBITS_INDEX> bci;
					bci.int_v = r | (b<<32);
					bci.s.color = colors[r].h();
					th_codes[idx_code] = bci;
					idx_code++;
				}
			}
		}
	}

	return idx_code;
}

void PVParallelView::PVZoneTreeNoAlloc::process_sse(PVZoneProcessing const& zp)
{
	const PVRow nrows = zp.nrows();
	_tree.resize(nrows);
	// Naive processing
	const uint32_t* pcol_a = zp.get_plotted_col_a();
	const uint32_t* pcol_b = zp.get_plotted_col_b();
	__m128i sse_y1, sse_y2, sse_bcodes;
	const PVRow nrows_sse = (nrows/4)*4;
	for (PVRow r = 0; r < nrows_sse; r += 4) {
		sse_y1 = _mm_load_si128((const __m128i*) &pcol_a[r]);
		sse_y2 = _mm_load_si128((const __m128i*) &pcol_b[r]);

		sse_y1 = _mm_srli_epi32(sse_y1, 32-NBITS_INDEX);
		sse_y2 = _mm_srli_epi32(sse_y2, 32-NBITS_INDEX);
		sse_bcodes = _mm_or_si128(sse_y1, _mm_slli_epi32(sse_y2, NBITS_INDEX));

		uint32_t b0 = _mm_extract_epi32(sse_bcodes, 0);
		if (_tree.push(b0, r+0)) {
			_first_elts[b0] = (r+0);
		}

		uint32_t b1 = _mm_extract_epi32(sse_bcodes, 1);
		if (_tree.push(b1, r+1)) {
			_first_elts[b1] = (r+1);
		}

		uint32_t b2 = _mm_extract_epi32(sse_bcodes, 2);
		if (_tree.push(b2, r+2)) {
			_first_elts[b2] = (r+2);
		}

		uint32_t b3 = _mm_extract_epi32(sse_bcodes, 3);
		if (_tree.push(b3, r+3)) {
			_first_elts[b3] = (r+3);
		}
	}
	for (PVRow r = nrows_sse; r < nrows; r++) {
		uint32_t y1 = pcol_a[r];
		uint32_t y2 = pcol_b[r];

		PVBCode b;
		b.int_v = 0;
		b.s.l = y1 >> (32-NBITS_INDEX);
		b.s.r = y2 >> (32-NBITS_INDEX);
		
		if (_tree.push(b.int_v, r)) {
			_first_elts[b.int_v] = r;
		}
	}
}

void PVParallelView::PVZoneTreeNoAlloc::process_omp_sse(PVZoneProcessing const& zp)
{
	// Naive processing
	const uint32_t* pcol_a = zp.get_plotted_col_a();
	const uint32_t* pcol_b = zp.get_plotted_col_b();
	const PVRow nrows = zp.nrows();
	tbb::tick_count start,end,red_start;
	Tree* thread_trees;
	uint32_t** thread_first_elts;
	int ntrees;
	const size_t nthreads = PVCore::PVHardwareConcurrency::get_physical_core_number();
#pragma omp parallel num_threads(nthreads)
	{
#pragma omp master
		{
			ntrees = omp_get_num_threads()-1;
			thread_trees = new Tree[ntrees];
			thread_first_elts = new uint32_t*[ntrees];
			for (int i=0; i<ntrees; i++) {
				thread_first_elts[i] = new PVRow[NBUCKETS];
				memset(thread_first_elts[i], PVROW_INVALID_VALUE, sizeof(PVRow)*NBUCKETS);
			}
		}
#pragma omp barrier

		// Initialize one tree per thread
		Tree* thread_tree = &thread_trees[omp_get_thread_num()-1];
		uint32_t* first_elts = thread_first_elts[omp_get_thread_num()-1];
#pragma omp master
		{
			thread_tree = &_tree;
			first_elts = _first_elts;
		}
		thread_tree->resize(nrows/omp_get_num_threads() + omp_get_num_threads());
		PVRow nrows_sse = (nrows/4)*4;
#pragma omp barrier
#pragma omp master
		{
			start = tbb::tick_count::now();
		}
		__m128i sse_y1, sse_y2, sse_bcodes;
#pragma omp for
		for (PVRow r = 0; r < nrows_sse; r += 4) {
			sse_y1 = _mm_load_si128((const __m128i*) &pcol_a[r]);
			sse_y2 = _mm_load_si128((const __m128i*) &pcol_b[r]);

			sse_y1 = _mm_srli_epi32(sse_y1, 32-NBITS_INDEX);
			sse_y2 = _mm_srli_epi32(sse_y2, 32-NBITS_INDEX);
			sse_bcodes = _mm_or_si128(sse_y1, _mm_slli_epi32(sse_y2, NBITS_INDEX));

			uint32_t b0 = _mm_extract_epi32(sse_bcodes, 0);
			if(thread_tree->push(b0, r+0)) {
				first_elts[b0] = r+0;
			}

			uint32_t b1 = _mm_extract_epi32(sse_bcodes, 1);
			if(thread_tree->push(b1, r+1)) {
				first_elts[b1] = r+1;
			}

			uint32_t b2 = _mm_extract_epi32(sse_bcodes, 2);
			if(thread_tree->push(b2, r+2)) {
				first_elts[b2] = r+2;
			}

			uint32_t b3 = _mm_extract_epi32(sse_bcodes, 3);
			if(thread_tree->push(b3, r+3)) {
				first_elts[b3] = r+3;
			}
		}
#pragma omp master
		for (PVRow r = nrows_sse; r < nrows; r++) {
			uint32_t y1 = pcol_a[r];
			uint32_t y2 = pcol_b[r];

			PVBCode b;
			b.int_v = 0;
			b.s.l = y1 >> (32-NBITS_INDEX);
			b.s.r = y2 >> (32-NBITS_INDEX);

			if(thread_tree->push(b.int_v, r)) {
				first_elts[b.int_v] = r;
			}
		}
	}

	red_start = tbb::tick_count::now();
#pragma omp parallel for num_threads(nthreads)
	for (ssize_t b = 0; b < (ssize_t) NBUCKETS; b++) {
		for (int ith = 0; ith < ntrees; ith++) {
			_tree.move_branch(b, b, thread_trees[ith]);
			_first_elts[b] = picviz_min(
				_first_elts[b],
				thread_first_elts[ith][b]
			);
		}
	}
	end = tbb::tick_count::now();
	for (int ith = 0; ith < ntrees; ith++) {
		_tree.take_buf(thread_trees[ith]);
		delete [] thread_first_elts[ith];
	}
	delete [] thread_first_elts;
	delete [] thread_trees;
	
	//PVLOG_INFO("OMP tree process in %0.4f ms.\n", (end-start).seconds()*1000.0);
	//PVLOG_INFO("OMP tree process reduction in %0.4f ms.\n", (end-red_start).seconds()*1000.0);
}

void PVParallelView::PVZoneTreeNoAlloc::get_float_pts(pts_t& pts, Picviz::PVPlotted::plotted_table_t const& org_plotted, PVRow nrows, PVCol col_a, PVCol col_b)
{
	pts.reserve(NBUCKETS*4);
	for (uint32_t i = 0; i < NBUCKETS; i++) {
		if (_tree.branch_valid(i)) {
			PVRow idx_first = _tree.get_first_elt_of_branch(i);
			pts.push_back(0.0f);
			pts.push_back(org_plotted[col_a*nrows+idx_first]);
			pts.push_back(1.0f);
			pts.push_back(org_plotted[col_b*nrows+idx_first]);
		}
	}
}

void PVParallelView::PVZoneTreeNoAlloc::filter_by_sel_tbb(Picviz::PVSelection const& sel)
{
	// returns a zone tree with only the selected lines
	BENCH_START(subtree);
	Picviz::PVSelection::const_pointer sel_buf = sel.get_buffer();
	TLS tls;
	const size_t nthreads = PVCore::PVHardwareConcurrency::get_physical_core_number();
	tbb::task_scheduler_init init(nthreads);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, NBUCKETS, GRAINSIZE), __impl::TBBPF(this, sel_buf, &tls), tbb::simple_partitioner());
	BENCH_END(subtree, "tbb::parallel_for", 1, 1, sizeof(PVRow), NBUCKETS);
}

void PVParallelView::PVZoneTreeNoAlloc::filter_by_sel_omp(Picviz::PVSelection const& sel)
{
	BENCH_START(subtree);
	Picviz::PVSelection::const_pointer sel_buf = sel.get_buffer();
	const size_t nthreads = omp_get_max_threads()/2;
#pragma omp parallel for schedule(dynamic, atol(getenv("GRAINSIZE"))) firstprivate(sel_buf) num_threads(nthreads)
	for (uint64_t b = 0; b < NBUCKETS; b++) {
		if (branch_valid(b)) {
			PVRow r = get_first_elt_of_branch(b);
			bool found = false;
			if ((sel_buf[r>>5]) & (1U<<(r&31))) {
				found = true;
			}
			else {
				Tree::const_branch_iterator it_src = _tree.begin_branch(b);
				it_src++;
				for (; it_src != _tree.end_branch(b); it_src++) {
					r = *(it_src);
					if ((sel_buf[r>>5]) & (1U<<(r&31))) {
						found = true;
						break;
					}
				}
			}
			if (found) {
				_sel_elts[b] = r;
			}
			else {
				_sel_elts[b] = PVROW_INVALID_VALUE;
			}
		}
	}
	BENCH_END(subtree, "filter_by_sel", 1, 1, sizeof(PVRow), NBUCKETS);
}

