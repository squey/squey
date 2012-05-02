#include <pvkernel/core/general.h>
#include <pvparallelview/PVZoneTree.h>
#include <pvparallelview/simple_lines_float_view.h>
#include <pvkernel/core/picviz_intrin.h>

#include <cassert>

#include <QMainWindow>

#include <omp.h>

void PVParallelView::PVZoneTreeBase::set_trans_plotted(plotted_int_t const& plotted, PVRow nrows, PVCol ncols)
{
	_plotted = &plotted;
	_ncols = ncols;
	_nrows = nrows;
	_nrows_aligned = ((_nrows+3)/4)*4;

	/*for (PVRow i = 0; i < NBUCKETS; i++) {
		_tree[i].reserve(_nrows);
	}*/
}

void PVParallelView::PVZoneTreeBase::display(QString const& name, Picviz::PVPlotted::plotted_table_t const& org_plotted)
{
	QMainWindow *window = new QMainWindow();
	window->setWindowTitle(name);
	SLFloatView *v = new SLFloatView(window);

	v->set_size(1024, 1024);
	v->set_ortho(1.0f, 1.0f);

	pts_t *pts = new pts_t();
	get_float_pts(*pts, org_plotted);
	PVLOG_INFO("Nb lines: %u\n", pts->size()/4);
	v->set_points(*pts);

	window->setCentralWidget(v);
	window->resize(v->sizeHint());
	window->show();
}

void PVParallelView::PVZoneTreeNoAlloc::process_sse()
{
	_tree.resize(_nrows);
	// Naive processing
	const uint32_t* pcol_a = get_plotted_col(_col_a);
	const uint32_t* pcol_b = get_plotted_col(_col_b);
	__m128i sse_y1, sse_y2, sse_bcodes;
	const PVRow nrows_sse = (_nrows/4)*4;
	for (PVRow r = 0; r < nrows_sse; r += 4) {
		sse_y1 = _mm_load_si128((const __m128i*) &pcol_a[r]);
		sse_y2 = _mm_load_si128((const __m128i*) &pcol_b[r]);

		sse_y1 = _mm_srli_epi32(sse_y1, 32-NBITS_INDEX);
		sse_y2 = _mm_srli_epi32(sse_y2, 32-NBITS_INDEX);
		sse_bcodes = _mm_or_si128(sse_y1, _mm_slli_epi32(sse_y2, NBITS_INDEX));

		_tree.push(_mm_extract_epi32(sse_bcodes, 0), r+0);
		_tree.push(_mm_extract_epi32(sse_bcodes, 1), r+1);
		_tree.push(_mm_extract_epi32(sse_bcodes, 2), r+2);
		_tree.push(_mm_extract_epi32(sse_bcodes, 3), r+3);
	}
	for (PVRow r = nrows_sse; r < _nrows; r++) {
		uint32_t y1 = pcol_a[r];
		uint32_t y2 = pcol_b[r];

		PVBCode b;
		b.int_v = 0;
		b.s.l = y1 >> (32-NBITS_INDEX);
		b.s.r = y2 >> (32-NBITS_INDEX);
		
		_tree.push(b.int_v, r);
	}
}

void PVParallelView::PVZoneTreeNoAlloc::process_omp_sse()
{
	// Naive processing
	const uint32_t* pcol_a = get_plotted_col(_col_a);
	const uint32_t* pcol_b = get_plotted_col(_col_b);
	tbb::tick_count start,end,red_start;
	Tree* thread_trees;
	int ntrees;
#pragma omp parallel
	{
#pragma omp master
		{
			ntrees = omp_get_num_threads()-1;
			thread_trees = new Tree[ntrees];
		}
#pragma omp barrier

		// Initialize one tree per thread
		Tree* thread_tree = &thread_trees[omp_get_thread_num()-1];
#pragma omp master
		{
			thread_tree = &_tree;
		}
		thread_tree->resize(_nrows/omp_get_num_threads() + omp_get_num_threads());
		PVRow nrows_sse = (_nrows/4)*4;
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

			thread_tree->push(_mm_extract_epi32(sse_bcodes, 0), r+0);
			thread_tree->push(_mm_extract_epi32(sse_bcodes, 1), r+1);
			thread_tree->push(_mm_extract_epi32(sse_bcodes, 2), r+2);
			thread_tree->push(_mm_extract_epi32(sse_bcodes, 3), r+3);
		}
#pragma omp master
		for (PVRow r = nrows_sse; r < _nrows; r++) {
			uint32_t y1 = pcol_a[r];
			uint32_t y2 = pcol_b[r];

			PVBCode b;
			b.int_v = 0;
			b.s.l = y1 >> (32-NBITS_INDEX);
			b.s.r = y2 >> (32-NBITS_INDEX);

			thread_tree->push(b.int_v, r);
		}
	}

	red_start = tbb::tick_count::now();
#pragma omp parallel for
	for (ssize_t b = 0; b < (ssize_t) NBUCKETS; b++) {
		for (int ith = 0; ith < ntrees; ith++) {
			_tree.move_branch(b, b, thread_trees[ith]);
		}
	}
	end = tbb::tick_count::now();
	for (int ith = 0; ith < ntrees; ith++) {
		_tree.take_buf(thread_trees[ith]);
	}
	delete [] thread_trees;
	
	//PVLOG_INFO("OMP tree process in %0.4f ms.\n", (end-start).seconds()*1000.0);
	//PVLOG_INFO("OMP tree process reduction in %0.4f ms.\n", (end-red_start).seconds()*1000.0);
}

size_t PVParallelView::PVZoneTreeNoAlloc::browse_tree_bci(PVHSVColor* colors, PVBCICode* codes)
{
	size_t idx_code = 0;
//#pragma omp parallel for reduction(+:idx_code) num_threads(4)
	for (uint64_t b = 0; b < NBUCKETS; b+=2) {
		const bool b1 = _tree.branch_valid(b);
		const bool b2 = _tree.branch_valid(b+1);
		if (b1 & b2) {
			PVRow const& r0 = _tree.get_first_elt_of_branch(b);
			PVRow r1 = _tree.get_first_elt_of_branch(b+1);
			//_mm_prefetch((const void*) &_tree.get_first_elt_of_branch((b+2)%NBUCKETS), _MM_HINT_NTA);
			//_mm_prefetch((const void*) &_tree.get_first_elt_of_branch((b+3)%NBUCKETS), _MM_HINT_NTA);

			// Load
			__m128i sse_bci_codes = _mm_loadl_epi64((__m128i const*) &r0);
			_mm_insert_epi64(sse_bci_codes, r1, 1);

			//  +------------+------------++------------+------------+
			//  |          0 | index (r1) ||          0 | index (r0) | (sse_bci_codes)
			//  +------------+------------++------------+------------+

			__m128i sse_lr;
			_mm_insert_epi64(sse_lr, b, 0);
			_mm_insert_epi64(sse_lr, b+1, 1);

			//  +------------+------------++------------+------------+
			//  |          0 |        lr1 ||          0 |        lr0 | (sse_lr)
			//  +------------+------------++------------+------------+

			__m128i sse_color = _mm_set1_epi32(0);
			_mm_insert_epi64(sse_color, colors[r0].h(), 0);
			_mm_insert_epi64(sse_color, colors[r1].h(), 1);
			sse_color = _mm_slli_epi64(sse_color, NBITS_INDEX*2);

			//  +------------+------------++------------+------------+
			//  |          0 |color1 << 20||          0 |color0 << 20| (sse_color)
			//  +------------+------------++------------+------------+

			__m128i sse_lrcolor;
			sse_lrcolor = _mm_or_si128(sse_color, sse_lr);
			sse_lrcolor = _mm_slli_epi64(sse_color, 32);

			//  +------------+------------++------------+------------+
			//  |   lrcolor1 |   0        ||   lrcolor0 |          0 | (sse_lrcolor)
			//  +------------+------------++------------+------------+

			sse_bci_codes = _mm_or_si128(sse_bci_codes, sse_lrcolor);

			//  +------------+------------++------------+------------+
			//  |   lrcolor1 | index (r1) ||   lrcolor0 | index (r0) | (sse_bci_codes)
			//  +------------+------------++------------+------------+

			if ((idx_code & 1) == 0) {
				_mm_store_si128((__m128i*)&codes[b], sse_bci_codes);
			}
			else {
				_mm_storeu_si128((__m128i*)&codes[b], sse_bci_codes);
			}

			idx_code += 2;
		}
		else
		if (b1 | b2) {
			uint64_t b0 = b + b2;
			PVRow r0 = _tree.get_first_elt_of_branch(b0);

			PVBCICode bci0;
			bci0.int_v = r0 | (b0<<32);
			bci0.s.color = colors[r0].h();
			codes[b] = bci0;
			idx_code++;
		}
	}

	return idx_code;
}

void PVParallelView::PVZoneTreeNoAlloc::get_float_pts(pts_t& pts, Picviz::PVPlotted::plotted_table_t const& org_plotted)
{
	pts.reserve(NBUCKETS*4);
	for (uint32_t i = 0; i < NBUCKETS; i++) {
		if (_tree.branch_valid(i)) {
			PVRow idx_first = _tree.get_first_elt_of_branch(i);
			pts.push_back(0.0f);
			pts.push_back(org_plotted[_col_a*_nrows+idx_first]);
			pts.push_back(1.0f);
			pts.push_back(org_plotted[_col_b*_nrows+idx_first]);
		}
	}
}
