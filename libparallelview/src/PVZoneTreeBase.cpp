/**
 * \file PVZoneTreeBase.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/general.h>
#include <pvkernel/core/picviz_intrin.h>

#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVZoneTree.h>
#include <pvparallelview/simple_lines_float_view.h>

#include <cassert>

#include <QMainWindow>

#include <omp.h>

PVParallelView::PVZoneTreeBase::PVZoneTreeBase()
{
	memset(_first_elts, PVROW_INVALID_VALUE, sizeof(PVRow)*NBUCKETS);
	memset(_sel_elts, PVROW_INVALID_VALUE, sizeof(PVRow)*NBUCKETS);
}

void PVParallelView::PVZoneTreeBase::display(QString const& name, Picviz::PVPlotted::plotted_table_t const& org_plotted, PVRow nrows, PVCol col_a, PVCol col_b)
{
	QMainWindow *window = new QMainWindow();
	window->setWindowTitle(name);
	SLFloatView *v = new SLFloatView(window);

	v->set_size(1024, 1024);
	v->set_ortho(1.0f, 1.0f);

	pts_t *pts = new pts_t();
	get_float_pts(*pts, org_plotted, nrows, col_a, col_b);
	PVLOG_INFO("Nb lines: %u\n", pts->size()/4);
	v->set_points(*pts);

	window->setCentralWidget(v);
	window->resize(v->sizeHint());
	window->show();
}

size_t PVParallelView::PVZoneTreeBase::browse_tree_bci_no_sse(PVHSVColor const* colors, PVBCICode<NBITS_INDEX>* codes) const
{
	size_t idx_code = 0;
//#pragma omp parallel for reduction(+:idx_code) num_threads(4)
	for (uint64_t b = 0; b < NBUCKETS; b++) {
		if (branch_valid(b)) {
			PVRow r = get_first_elt_of_branch(b);
			PVBCICode<NBITS_INDEX> bci;
			bci.int_v = r | (b<<32);
			bci.s.color = colors[r].h();
			codes[idx_code] = bci;
			idx_code++;
		}
	}

	return idx_code;
}

size_t PVParallelView::PVZoneTreeBase::browse_tree_bci_old(PVHSVColor const* colors, PVBCICode<NBITS_INDEX>* codes) const
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

			PVBCICode<NBITS_INDEX> bci0;
			bci0.int_v = r0 | (b0<<32);
			bci0.s.color = colors[r0].h();
			codes[idx_code] = bci0;
			idx_code++;
		}
	}

	return idx_code;
}

size_t PVParallelView::PVZoneTreeBase::browse_tree_bci(PVHSVColor const* colors, PVBCICode<NBITS_INDEX>* codes) const
{
	return browse_tree_bci_from_buffer(_first_elts, colors, codes);
}

size_t PVParallelView::PVZoneTreeBase::browse_tree_bci_sel(PVHSVColor const* colors, PVBCICode<NBITS_INDEX>* codes) const
{
	return browse_tree_bci_from_buffer(_sel_elts, colors, codes);
}

size_t PVParallelView::PVZoneTreeBase::browse_tree_bci_from_buffer(const PVRow* elts, PVHSVColor const* colors, PVBCICode<NBITS_INDEX>* codes) const
{
	size_t idx_code = 0;

//	const size_t nthreads = atol(getenv("NUM_THREADS"));/*omp_get_max_threads()/2;*/
//#pragma omp parallel num_threads(nthreads)
//	{
	//PVBCICode<NBITS_INDEX>* th_codes = PVBCICode::allocate_codes(NBUCKETS);
//#pragma omp for reduction(+:idx_code) schedule(dynamic, 6)
	for (uint64_t b = 0; b < NBUCKETS; b+=4) {

		__m128i sse_ff = _mm_set1_epi32(0xFFFFFFFF);
		__m128i sse_index = _mm_load_si128((const __m128i*) &elts[b]);
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
				PVRow r = elts[b0];
				if (r != PVROW_INVALID_VALUE){
					PVBCICode<NBITS_INDEX> bci;
					bci.int_v = r | (b0<<32);
					bci.s.color = colors[r].h();
					codes[idx_code] = bci;
					idx_code++;
				}
			}
		}
	}
//	}

	return idx_code;
}
