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

	memset(_first_elts, INVALID_VALUE, sizeof(PVRow)*NBUCKETS);
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

size_t PVParallelView::PVZoneTreeBase::browse_tree_bci_no_sse(PVHSVColor* colors, PVBCICode* codes)
{
	size_t idx_code = 0;
//#pragma omp parallel for reduction(+:idx_code) num_threads(4)
	for (uint64_t b = 0; b < NBUCKETS; b++) {
		if (branch_valid(b)) {
			PVRow r = get_first_elt_of_branch(b);
			PVBCICode bci;
			bci.int_v = r | (b<<32);
			bci.s.color = colors[r].h();
			codes[idx_code] = bci;
			idx_code++;
		}
	}

	return idx_code;
}

size_t PVParallelView::PVZoneTreeBase::browse_tree_bci_old(PVHSVColor* colors, PVBCICode* codes)
{
	size_t idx_code = 0;
//#pragma omp parallel for reduction(+:idx_code) num_threads(4)
	for (uint64_t b = 0; b < NBUCKETS; b+=2) {
		const bool b1 = branch_valid(b+0);
		const bool b2 = branch_valid(b+1);
		if (b1 & b2) {
			bool b3 = false;
			bool b4 = false;
			if (b < NBUCKETS-2) {
				b3 = branch_valid(b+2);
				b4 = branch_valid(b+3);
			}
			PVRow const& r0 = get_first_elt_of_branch(b);
			PVRow r1 = get_first_elt_of_branch(b+1);
			if (b3 & b4) {
				PVRow r2 = get_first_elt_of_branch(b+2);
				PVRow r3 = get_first_elt_of_branch(b+3);

				__m128i sse_lr;
				sse_lr = _mm_insert_epi32(sse_lr, b+0, 0);
				sse_lr = _mm_insert_epi32(sse_lr, b+1, 1);
				sse_lr = _mm_insert_epi32(sse_lr, b+2, 2);
				sse_lr = _mm_insert_epi32(sse_lr, b+3, 3);

				//  +------------+------------++------------+------------+
				//  |        lr3 |        lr2 ||        lr1 |        lr0 | (sse_lr)
				//  +------------+------------++------------+------------+

				__m128i sse_color;
				sse_color = _mm_insert_epi32(sse_color, colors[r0].h(), 0);
				sse_color = _mm_insert_epi32(sse_color, colors[r1].h(), 1);
				sse_color = _mm_insert_epi32(sse_color, colors[r2].h(), 2);
				sse_color = _mm_insert_epi32(sse_color, colors[r3].h(), 3);
				sse_color = _mm_slli_epi32(sse_color, NBITS_INDEX*2);

				//  +------------+------------++------------+------------+
				//  |color3 << 20|color2 << 20||color1 << 20|color0 << 20| (sse_color)
				//  +------------+------------++------------+------------+

				__m128i sse_lrcolor;
				sse_lrcolor = _mm_or_si128(sse_color, sse_lr);

				//  +------------+------------++------------+------------+
				//  |   lrcolor3 |   lrcolor2 ||   lrcolor1 |   lrcolor0 | (sse_lrcolor)
				//  +------------+------------++------------+------------+

				__m128i sse_index;
				sse_index = _mm_insert_epi32(sse_index, r0, 0);
				sse_index = _mm_insert_epi32(sse_index, r1, 1);
				sse_index = _mm_insert_epi32(sse_index, r2, 2);
				sse_index = _mm_insert_epi32(sse_index, r3, 3);

				//  +------------+------------++------------+------------+
				//  | index (r3) | index (r2) || index (r1) | index (r0) | (sse_index)
				//  +------------+------------++------------+------------+

				__m128i sse_bcicodes0_1 = _mm_unpacklo_epi32(sse_index, sse_lrcolor);
				__m128i sse_bcicodes2_3 = _mm_unpackhi_epi32(sse_index, sse_lrcolor);

				//  +------------+------------++------------+------------+
				//  |   lrcolor1 | index (r1) ||   lrcolor0 | index (r0) | (sse_bcicodes0_1)
				//  +------------+------------++------------+------------+
				//  +------------+------------++------------+------------+
				//  |   lrcolor3 | index (r3) ||   lrcolor2 | index (r2) | (sse_bcicodes2_3)
				//  +------------+------------++------------+------------+


				if ((idx_code & 1) == 0) {
					_mm_stream_si128((__m128i*)&codes[idx_code+0], sse_bcicodes0_1);
					_mm_stream_si128((__m128i*)&codes[idx_code+2], sse_bcicodes2_3);
				}
				else {
					_mm_storeu_si128((__m128i*)&codes[idx_code+0], sse_bcicodes0_1);
					_mm_storeu_si128((__m128i*)&codes[idx_code+2], sse_bcicodes2_3);
				}

				idx_code += 4;
				b += 2;

			}
			else {
				//_mm_prefetch((const void*) &_tree.get_first_elt_of_branch((b+2)%NBUCKETS), _MM_HINT_NTA);
				//_mm_prefetch((const void*) &_tree.get_first_elt_of_branch((b+3)%NBUCKETS), _MM_HINT_NTA);

				// Load
				__m128i sse_bci_codes = _mm_loadl_epi64((__m128i const*) &r0);
				sse_bci_codes = _mm_insert_epi64(sse_bci_codes, r1, 1);

				//  +------------+------------++------------+------------+
				//  |          0 | index (r1) ||          0 | index (r0) | (sse_bci_codes)
				//  +------------+------------++------------+------------+

				__m128i sse_lr;
				sse_lr = _mm_insert_epi64(sse_lr, b+0, 0);
				sse_lr = _mm_insert_epi64(sse_lr, b+1, 1);

				//  +------------+------------++------------+------------+
				//  |          0 |        lr1 ||          0 |        lr0 | (sse_lr)
				//  +------------+------------++------------+------------+

				__m128i sse_color = _mm_set1_epi32(0);
				sse_color = _mm_insert_epi64(sse_color, colors[r0].h(), 0);
				sse_color = _mm_insert_epi64(sse_color, colors[r1].h(), 1);
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
					_mm_stream_si128((__m128i*)&codes[idx_code], sse_bci_codes);
				}
				else {
					_mm_storeu_si128((__m128i*)&codes[idx_code], sse_bci_codes);
				}

				idx_code += 2;
			}
		}
		else if (b1 | b2) {
			uint64_t b0 = b + b2;
			PVRow r0 = get_first_elt_of_branch(b0);

			PVBCICode bci0;
			bci0.int_v = r0 | (b0<<32);
			bci0.s.color = colors[r0].h();
			codes[idx_code] = bci0;
			idx_code++;
		}
	}

	return idx_code;
}

size_t PVParallelView::PVZoneTreeBase::browse_tree_bci(PVHSVColor* colors, PVBCICode* codes)
{
	size_t idx_code = 0;

//#pragma omp parallel for reduction(+:idx_code) num_threads(4)
	for (uint64_t b = 0; b < NBUCKETS; b+=4) {

		__m128i sse_ff = _mm_set1_epi32(0xFFFFFFFF);
		__m128i sse_index = _mm_load_si128((const __m128i*) &_first_elts[b]);
		__m128i see_cmp = _mm_cmpeq_epi32(sse_ff, sse_index);

		if (_mm_testz_si128(see_cmp, sse_ff)) {

				__m128i sse_lr;
				sse_lr = _mm_insert_epi32(sse_lr, b+0, 0);
				sse_lr = _mm_insert_epi32(sse_lr, b+1, 1);
				sse_lr = _mm_insert_epi32(sse_lr, b+2, 2);
				sse_lr = _mm_insert_epi32(sse_lr, b+3, 3);

				//  +------------+------------++------------+------------+
				//  |        lr3 |        lr2 ||        lr1 |        lr0 | (sse_lr)
				//  +------------+------------++------------+------------+

				__m128i sse_color;
				sse_color = _mm_insert_epi32(sse_color, colors[_mm_extract_epi32(sse_index, 0)].h(), 0);
				sse_color = _mm_insert_epi32(sse_color, colors[_mm_extract_epi32(sse_index, 1)].h(), 1);
				sse_color = _mm_insert_epi32(sse_color, colors[_mm_extract_epi32(sse_index, 2)].h(), 2);
				sse_color = _mm_insert_epi32(sse_color, colors[_mm_extract_epi32(sse_index, 3)].h(), 3);
				sse_color = _mm_slli_epi32(sse_color, NBITS_INDEX*2);

				//  +------------+------------++------------+------------+
				//  |color3 << 20|color2 << 20||color1 << 20|color0 << 20| (sse_color)
				//  +------------+------------++------------+------------+

				__m128i sse_lrcolor;
				sse_lrcolor = _mm_or_si128(sse_color, sse_lr);

				//  +------------+------------++------------+------------+
				//  |   lrcolor3 |   lrcolor2 ||   lrcolor1 |   lrcolor0 | (sse_lrcolor)
				//  +------------+------------++------------+------------+

				__m128i sse_bcicodes0_1 = _mm_unpacklo_epi32(sse_index, sse_lrcolor);
				__m128i sse_bcicodes2_3 = _mm_unpackhi_epi32(sse_index, sse_lrcolor);

				//  +------------+------------++------------+------------+
				//  |   lrcolor1 | index (r1) ||   lrcolor0 | index (r0) | (sse_bcicodes0_1)
				//  +------------+------------++------------+------------+
				//  +------------+------------++------------+------------+
				//  |   lrcolor3 | index (r3) ||   lrcolor2 | index (r2) | (sse_bcicodes2_3)
				//  +------------+------------++------------+------------+


				if ((idx_code & 1) == 0) {
					_mm_stream_si128((__m128i*)&codes[idx_code+0], sse_bcicodes0_1);
					_mm_stream_si128((__m128i*)&codes[idx_code+2], sse_bcicodes2_3);
				}
				else {
					_mm_storeu_si128((__m128i*)&codes[idx_code+0], sse_bcicodes0_1);
					_mm_storeu_si128((__m128i*)&codes[idx_code+2], sse_bcicodes2_3);
				}

				idx_code += 4;
			}
		else {
			for (int i=0; i<4; i++) {
				uint64_t b0 = b + i;
				PVRow r = get_first_elt_of_branch(b0);
				if (branch_valid(b0)){
					PVBCICode bci;
					bci.int_v = r | (b0<<32);
					bci.s.color = colors[r].h();
					codes[idx_code] = bci;
					idx_code++;
				}
			}
		}
	}

	return idx_code;
}

size_t PVParallelView::PVZoneTreeNoAlloc::browse_tree_bci_by_sel(PVHSVColor* colors, PVBCICode* codes, Picviz::PVSelection const& sel)
{
	size_t idx_code = 0;
	Picviz::PVSelection::const_pointer sel_buf = sel.get_buffer();
#pragma omp parallel firstprivate(sel_buf) reduction(+:idx_code) num_threads(4)
	{
		PVBCICode* th_codes = PVBCICode::allocate_codes(NBUCKETS);
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
					PVBCICode bci;
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
	for (PVRow r = nrows_sse; r < _nrows; r++) {
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

void PVParallelView::PVZoneTreeNoAlloc::process_omp_sse()
{
	// Naive processing
	const uint32_t* pcol_a = get_plotted_col(_col_a);
	const uint32_t* pcol_b = get_plotted_col(_col_b);
	tbb::tick_count start,end,red_start;
	Tree* thread_trees;
	uint32_t** thread_first_elts;
	int ntrees;
#pragma omp parallel num_threads(64)
	{
#pragma omp master
		{
			ntrees = omp_get_num_threads()-1;
			thread_trees = new Tree[ntrees];
			thread_first_elts = new uint32_t*[ntrees];
			for (int i=0; i<ntrees; i++) {
				thread_first_elts[i] = new PVRow[NBUCKETS];
				memset(thread_first_elts[i], INVALID_VALUE, sizeof(PVRow)*NBUCKETS);
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
		for (PVRow r = nrows_sse; r < _nrows; r++) {
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
#pragma omp parallel for
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
