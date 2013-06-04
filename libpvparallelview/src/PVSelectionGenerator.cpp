/**
 * \file PVSelectionGenerator.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/picviz_bench.h>
#include <pvkernel/core/picviz_intrin.h>
#include <pvkernel/core/PVHardwareConcurrency.h>

#include <QApplication>

#include <picviz/PVSelection.h>

#include <pvparallelview/PVSelectionGenerator.h>
#include <pvparallelview/PVBCode.h>
#include <pvparallelview/PVZoneTree.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVAbstractAxisSlider.h>
#include <pvparallelview/PVHitGraphBlocksManager.h>
#include <pvparallelview/PVHitGraphSSEHelpers.h>

#include <tbb/enumerable_thread_specific.h>

#include <omp.h>

struct PVLineEqInt
{
	int a;
	int b;
	int c;
	inline int operator()(int X, int Y) const { return a*X+b*Y+c; }
};

uint32_t PVParallelView::PVSelectionGenerator::compute_selection_from_parallel_view_rect(
	PVLinesView& lines_view,
	PVZoneID zone_id,
	QRect rect,
	Picviz::PVSelection& sel
)
{
	uint32_t nb_selected = 0;

	int32_t width = lines_view.get_zone_width(zone_id);

	PVZoneTree const& ztree = lines_view.get_zones_manager().get_zone_tree<PVZoneTree>(zone_id);
	PVParallelView::PVBCode code_b;

	if (rect.isNull()) {
		return 0;
	}

	BENCH_START(compute_selection_from_parallel_view_rect);
	PVLineEqInt line;
	line.b = -width;

	//tbb::tag_tls_construct_args tag_c;
	//tbb::enumerable_thread_specific<Picviz::PVSelection> sel_tls(tag_c, Picviz::PVSelection::tag_allocate_empty());
	//tbb::enumerable_thread_specific<Picviz::PVSelection> sel_tls;
	for (uint32_t branch = 0 ; branch < NBUCKETS; branch++)
	{
		//PVRow r =  ztree.get_first_elt_of_branch(branch);
		//Picviz::PVSelection& sel_th = sel_tls.local();
		code_b.int_v = branch;
		int32_t y1 = code_b.s.l;
		int32_t y2 = code_b.s.r;

		line.a = y2 - y1;
		line.c = y1*width;

		const bool a = line(rect.topLeft().x(), rect.topLeft().y()) >= 0;
		const bool b = line(rect.topRight().x(), rect.topRight().y()) >=0;
		const bool c = line(rect.bottomLeft().x(), rect.bottomLeft().y()) >=0;
		const bool d = line(rect.bottomRight().x(), rect.bottomRight().y()) >=0;

		bool is_line_selected = (a | b | c | d) & (!(a & b & c & d));

		if (is_line_selected)
		{
			uint32_t branch_count = ztree.get_branch_count(branch);
			for (size_t i = 0; i < branch_count; i++) {
				const PVRow cur_r = ztree.get_branch_element(branch, i);
				//sel_th.set_bit_fast(cur_r);
				sel.set_bit_fast(cur_r);
//#pragma omp atomic
				//sel.get_buffer()[Picviz::PVSelection::line_index_to_chunk(cur_r)] |= 1UL<<Picviz::PVSelection::line_index_to_chunk_bit(cur_r);
			}
			nb_selected += branch_count;
		}
	}
	BENCH_END(compute_selection_from_parallel_view_rect, "compute_selection", sizeof(PVRow), NBUCKETS, 1, 1);

	/*
	if (nb_selected != 0) {
		BENCH_START(merge);
		decltype(sel_tls)::const_iterator it = sel_tls.begin();
		sel = *it;
		it++;
		for (; it != sel_tls.end(); it++) {
			sel.or_optimized(*it);
		}
		BENCH_END(merge, "compute_selection_merge", 1, 1, 1, 1);
	}*/

	return nb_selected;
}

uint32_t PVParallelView::PVSelectionGenerator::compute_selection_from_parallel_view_sliders(
	PVLinesView& lines_view,
	PVZoneID zone_id,
	const typename PVAxisGraphicsItem::selection_ranges_t& ranges,
	Picviz::PVSelection& sel)
{
	uint32_t nb_selected = 0;

	sel.select_none();

	PVParallelView::PVBCode code_b;

	if (zone_id < lines_view.get_zones_manager().get_number_of_managed_zones()) {
		// process the left side of zones
		PVZoneTree const& ztree = lines_view.get_zones_manager().get_zone_tree<PVZoneTree>(zone_id);

		for (auto &range : ranges) {
			uint64_t range_min = range.first;
			uint64_t range_max = range.second;

			uint32_t zt_min = range_min / BUCKET_ELT_COUNT;
			uint32_t zt_max = range_max / BUCKET_ELT_COUNT;

			uint64_t zt_range_min = zt_min * (uint64_t)BUCKET_ELT_COUNT;
			uint64_t zt_range_max = zt_max * (uint64_t)BUCKET_ELT_COUNT;

			bool need_zzt_min = false;
			uint32_t zzt_min_idx = 0;
			bool need_zzt_max = false;
			uint32_t zzt_max_idx = 0;

			if (zt_range_min != range_min) {
				zzt_min_idx = zt_min;
				++zt_min;
				need_zzt_min = true;
			}

			if (zt_range_max != range_max) {
				zzt_max_idx = zt_max;
				need_zzt_max = true;
			}

			if (zzt_max_idx == 1024) {
				need_zzt_max = false;
			}

			if (need_zzt_min && need_zzt_max && (zzt_min_idx == zzt_max_idx)) {
				/* it the two quadtrees must be traversed, make
				 * sure they are different, otherwise the same
				 * quadtree will be travers twice
				 */
				need_zzt_max = false;
			}

			for(PVRow t1 = zt_min; t1 < zt_max; ++t1) {
				for (PVRow t2 = 0; t2 < 1024; ++t2) {
					PVRow branch = (t2 * 1024) + t1;
					PVRow r = ztree.get_first_elt_of_branch(branch);
					if(r == PVROW_INVALID_VALUE) {
						continue;
					}

					code_b.int_v = branch;
					uint32_t y1 = code_b.s.l;
					bool is_line_selected = ((zt_min <= y1) && (y1 <= zt_max));

					if (is_line_selected == false) {
						continue;
					}

					//ztree._sel_elts[branch] = r;
					uint32_t branch_count = ztree.get_branch_count(branch);
					for (size_t i = 0; i < branch_count; ++i) {
						sel.set_bit_fast(ztree.get_branch_element(branch, i));
					}
					nb_selected += branch_count;
				}
			}

			if (need_zzt_min) {
				PVZoomedZoneTree const& zztree = lines_view.get_zones_manager().get_zone_tree<PVZoomedZoneTree>(zone_id);
				nb_selected += zztree.compute_selection_y1(zzt_min_idx, range_min,
				                                           range_max, sel);
			}

			if (need_zzt_max) {
				PVZoomedZoneTree const& zztree = lines_view.get_zones_manager().get_zone_tree<PVZoomedZoneTree>(zone_id);
				nb_selected += zztree.compute_selection_y1(zzt_max_idx, range_min,
				                                           range_max, sel);
			}
		}
	} else {
		// process the right side of zones
		PVZoneTree const& ztree = lines_view.get_zones_manager().get_zone_tree<PVZoneTree>(zone_id - 1);

		for (auto &range : ranges) {
			uint64_t range_min = range.first;
			uint64_t range_max = range.second;

			uint32_t zt_min = range_min / BUCKET_ELT_COUNT;
			uint32_t zt_max = range_max / BUCKET_ELT_COUNT;

			uint64_t zt_range_min = zt_min * (uint64_t)BUCKET_ELT_COUNT;
			uint64_t zt_range_max = zt_max * (uint64_t)BUCKET_ELT_COUNT;

			bool need_zzt_min = false;
			uint32_t zzt_min_idx = 0;
			bool need_zzt_max = false;
			uint32_t zzt_max_idx = 0;

			if (zt_range_min != range_min) {
				zzt_min_idx = zt_min;
				++zt_min;
				need_zzt_min = true;
			}

			if (zt_range_max != range_max) {
				zzt_max_idx = zt_max;
				need_zzt_max = true;
			}

			if (zzt_max_idx == 1024) {
				need_zzt_max = false;
			}

			if (need_zzt_min && need_zzt_max && (zzt_min_idx == zzt_max_idx)) {
				/* it the two quadtrees must be traversed, make
				 * sure they are different, otherwise the same
				 * quadtree will be travers twice
				 */
				need_zzt_max = false;
			}

			for (PVRow t1 = 0; t1 < 1024; ++t1) {
				for(PVRow t2 = zt_min; t2 < zt_max; ++t2) {
					PVRow branch = (t2 * 1024) + t1;
					PVRow r = ztree.get_first_elt_of_branch(branch);
					if(r == PVROW_INVALID_VALUE) {
						continue;
					}

					code_b.int_v = branch;
					uint32_t y2 = code_b.s.r;
					bool is_line_selected = ((zt_min <= y2) && (y2 <= zt_max));

					if (is_line_selected == false) {
						continue;
					}

					//ztree._sel_elts[branch] = r;
					uint32_t branch_count = ztree.get_branch_count(branch);
					for (size_t i = 0; i < branch_count; ++i) {
						sel.set_bit_fast(ztree.get_branch_element(branch, i));
					}
					nb_selected += branch_count;
				}
			}

			if (need_zzt_min) {
				PVZoomedZoneTree const& zztree = lines_view.get_zones_manager().get_zone_tree<PVZoomedZoneTree>(zone_id - 1);
				nb_selected += zztree.compute_selection_y2(zzt_min_idx, range_min,
				                                           range_max, sel);
			}

			if (need_zzt_max) {
				PVZoomedZoneTree const& zztree = lines_view.get_zones_manager().get_zone_tree<PVZoomedZoneTree>(zone_id - 1);
				nb_selected += zztree.compute_selection_y2(zzt_max_idx, range_min,
				                                           range_max, sel);
			}
		}
	}

	return nb_selected;
}

/*****************************************************************************
 * PVParallelView::PVSelectionGenerator::compute_selection_from_hit_count_view_rect
 *****************************************************************************/
uint32_t PVParallelView::PVSelectionGenerator::compute_selection_from_hit_count_view_rect(
	const PVHitGraphBlocksManager& manager,
	const QRectF& rect,
	const uint32_t max_count,
	Picviz::PVSelection& sel,
	bool use_selectable
)
{
	return __impl::compute_selection_from_hit_count_view_rect_sse_invariant_omp(manager, rect, max_count, sel, use_selectable);
}

/*****************************************************************************
 * PVParallelView::PVSelectionGenerator::compute_selection_from_hit_count_view_rect_serial
 *****************************************************************************/
uint32_t PVParallelView::__impl::compute_selection_from_hit_count_view_rect_serial(
	const PVHitGraphBlocksManager& manager,
	const QRectF& rect,
	const uint32_t max_count,
	Picviz::PVSelection& sel)
{
	sel.ensure_allocated();

	// The interval described here is of the type [a,b] (that is the maximum is taken into account)
	const uint32_t v_min = PVCore::clamp(floor(rect.top()), 0.0, (double)UINT32_MAX);
	const uint32_t v_max = PVCore::clamp(ceil(rect.bottom()), 0.0, (double)UINT32_MAX);

	// The interval described here is of the type [a,b[ (that is the maximum isn't taken into account)
	// "null" counted event can't be selected, so let's clamp c_min between [1,max_count]
	const uint32_t c_min = PVCore::clamp(ceil(rect.left()),  1.0, (double)(max_count+1));
	const uint32_t c_max = PVCore::clamp(ceil(rect.right()), 1.0, (double)(max_count+1));

	uint32_t nb_selected = 0;

	const uint32_t* plotted = manager.get_plotted();

	const uint32_t nrows = manager.get_nrows();

	BENCH_START(b);
	for(PVRow i = 0; i < nrows; ++i) {
		const uint32_t v = plotted[i];
		if ((v >= v_min) && (v <= v_max)) {
			const uint32_t c = manager.get_count_for(v);
			if ((c >= c_min) && (c < c_max)) {
				sel.set_bit_fast(i);
				++nb_selected;
				continue;
			}
		}
		sel.clear_bit_fast(i);
	}
	BENCH_END(b, "serial", nrows, sizeof(uint32_t), 1, 1);

	return nb_selected;
}

/*****************************************************************************
 * PVParallelView::__impl::compute_selection_from_hit_count_view_rect_serial_invariant
 *****************************************************************************/
uint32_t PVParallelView::__impl::compute_selection_from_hit_count_view_rect_serial_invariant(
	const PVHitGraphBlocksManager& manager,
	const QRectF& rect,
	const uint32_t max_count,
	Picviz::PVSelection& sel)
{
	sel.ensure_allocated();

	// The interval described here is of the type [a,b] (that is the maximum is taken into account)
	const uint32_t v_min = PVCore::clamp(floor(rect.top()), 0.0, (double)UINT32_MAX);
	const uint32_t v_max = PVCore::clamp(ceil(rect.bottom()), 0.0, (double)UINT32_MAX);

	// The interval described here is of the type [a,b[ (that is the maximum isn't taken into account)
	// "null" counted event can't be selected, so let's clamp c_min between [1,max_count]
	const uint32_t c_min = PVCore::clamp(ceil(rect.left()),  1.0, (double)(max_count+1));
	const uint32_t c_max = PVCore::clamp(ceil(rect.right()), 1.0, (double)(max_count+1));

	uint32_t nb_selected = 0;

	const uint32_t* plotted = manager.get_plotted();

	const uint32_t nrows = manager.get_nrows();

	const PVParallelView::PVHitGraphData& data = manager.hgdata();
	const int zoom = manager.last_zoom();
	const int nbits = data.nbits();
	const int idx_shift = (32 - nbits) - zoom;
	const uint32_t zoom_shift = 32 - zoom;
	const uint32_t zoom_mask = ((1ULL << zoom_shift) - 1ULL);
	const int32_t base_y = (uint64_t)(manager.last_y_min()) >> zoom_shift;
	const uint32_t y_min_ref = (uint64_t)base_y << zoom_shift;
	const int nblocks = data.nblocks();
	const double alpha = manager.last_alpha();

	BENCH_START(b);
	for(PVRow i = 0; i < nrows; ++i) {
		uint64_t v = plotted[i];
		if ((v >= v_min) && (v <= v_max)) {
			const int32_t base = v >> zoom_shift;

			int p = base - base_y;
			if ((p >= 0) && (p < nblocks)) {
				v = (v - y_min_ref) * alpha;
				const uint32_t idx = ((uint32_t)(v & zoom_mask)) >> idx_shift;
				const uint32_t c = data.buffer_all().buffer()[idx];

				if ((c >= c_min) && (c < c_max)) {
					sel.set_bit_fast(i);
					++nb_selected;
					continue;
				}
			}
		}
		sel.clear_bit_fast(i);
	}
	BENCH_END(b, "serial-invariant", nrows, sizeof(uint32_t), 1, 1);

	return nb_selected;
}

/*****************************************************************************
 * PVParallelView::__impl::compute_selection_from_hit_count_view_rect_sse
 *****************************************************************************/
uint32_t PVParallelView::__impl::compute_selection_from_hit_count_view_rect_sse(
	const PVHitGraphBlocksManager& manager,
	const QRectF& rect,
	const uint32_t max_count,
	Picviz::PVSelection& sel)
{
	sel.ensure_allocated();

	// The interval described here is of the type [a,b] (that is the maximum is taken into account)
	const uint32_t v_min = PVCore::clamp(floor(rect.top()), 0.0, (double)UINT32_MAX);
	const uint32_t v_max = PVCore::clamp(ceil(rect.bottom()), 0.0, (double)UINT32_MAX);

	// The interval described here is of the type [a,b[ (that is the maximum isn't taken into account)
	// "null" counted event can't be selected, so let's clamp c_min between [1,max_count]
	const uint32_t c_min = PVCore::clamp(ceil(rect.left()),  1.0, (double)(max_count+1));
	const uint32_t c_max = PVCore::clamp(ceil(rect.right()), 1.0, (double)(max_count+1));

	const __m128i v_min_sse = _mm_set1_epi32(v_min);
	const __m128i v_max_sse = _mm_set1_epi32(v_max);

	const __m128i c_min_sse = _mm_set1_epi32(c_min);
	const __m128i c_max_sse = _mm_set1_epi32(c_max);

	uint32_t nb_selected = 0;

	const uint32_t* plotted = manager.get_plotted();
	const uint32_t nrows = manager.get_nrows();
	const uint32_t nrows_sse = nrows & ~63U;
	BENCH_START(b);
	PVRow i;
	for (i = 0; i < nrows_sse; i += 64) {
		// Compute one chunk of the selection
		uint64_t chunk = 0;
		for (int j = 0; j < 64; j += 4) {
			const __m128i y_sse = _mm_load_si128((__m128i const*) &plotted[i+j]);
			// _mm_andnot_si128(a,b) = ~a & b
			// _mm_cmplt_epi32(a,b) = a < b;
			// thus andnot(cmplt(a,b),cmplt(a,c)) <=> (!(a < b)) && (a < c) <=> (a >=b) && (a < c)
			const __m128i mask_y = picviz_mm_cmprange_in_epu32(y_sse, v_min_sse, v_max_sse);

			if (!_mm_test_all_zeros(mask_y, _mm_set1_epi32(0xFFFFFFFFU))) {

				const __m128i count_sse = manager.get_count_for(y_sse);

				const __m128i mask_count = picviz_mm_cmprange_epi32(count_sse, c_min_sse, c_max_sse);

				const __m128i mask = _mm_and_si128(mask_y, mask_count);

				// Get selection bits and write them
				const uint64_t sel_bits = _mm_movemask_ps(reinterpret_cast<__m128>(mask));
				chunk |= sel_bits << j;
			}
		}
		sel.set_chunk_fast(Picviz::PVSelection::line_index_to_chunk(i), chunk);
	}
	for (; i < nrows; i++) {
		const uint32_t v = plotted[i];
		if ((v >= v_min) && (v < v_max)) {
			const uint32_t c = manager.get_count_for(v);
			if ((c >= c_min) && (c < c_max)) {
				sel.set_bit_fast(i);
				++nb_selected;
				continue;
			}
		}
		sel.clear_bit_fast(i);
	}

	BENCH_END(b, "sse", nrows, sizeof(uint32_t), 1, 1);

	return nb_selected;
}

/*****************************************************************************
 * PVParallelView::__impl::compute_selection_from_hit_count_view_rect_sse_invariant_omp
 *****************************************************************************/
uint32_t PVParallelView::__impl::compute_selection_from_hit_count_view_rect_sse_invariant_omp(
	const PVHitGraphBlocksManager& manager,
	const QRectF& rect,
	const uint32_t max_count,
	Picviz::PVSelection& sel,
	bool use_selectable
	)
{
	sel.ensure_allocated();

	// The interval described here is of the type [a,b] (that is the maximum is taken into account)
	const uint32_t v_min = PVCore::clamp(floor(rect.top()), 0.0, (double)UINT32_MAX);
	const uint32_t v_max = PVCore::clamp(ceil(rect.bottom()), 0.0, (double)UINT32_MAX);

	// The interval described here is of the type [a,b[ (that is the maximum isn't taken into account)
	// "null" counted event can't be selected, so let's clamp c_min between [1,max_count]
	const uint32_t c_min = PVCore::clamp(ceil(rect.left()),  1.0, (double)(max_count+1));
	const uint32_t c_max = PVCore::clamp(ceil(rect.right()), 1.0, (double)(max_count+1));

	const __m128i v_min_sse = _mm_set1_epi32(v_min);
	const __m128i v_max_sse = _mm_set1_epi32(v_max);

	const __m128i c_min_sse = _mm_set1_epi32(c_min);
	const __m128i c_max_sse = _mm_set1_epi32(c_max);

	const PVParallelView::PVHitGraphData& data = manager.hgdata();
	const int zoom = manager.last_zoom();
	const int nbits = data.nbits();
	const int idx_shift = (32 - nbits) - zoom;
	const uint32_t zoom_shift = 32 - zoom;
	const __m128i zoom_mask_sse = _mm_set1_epi32((1ULL << zoom_shift) - 1ULL);
	const __m128i base_y_sse = _mm_set1_epi32((uint64_t)(manager.last_y_min()) >> zoom_shift);
	const __m128i y_min_ref_sse = _mm_slli_epi32(base_y_sse, zoom_shift);
	const __m128i nblocks_sse = _mm_set1_epi32(data.nblocks());
#ifdef __AVX__
	const __m256d alpha_sse = _mm256_set1_pd(manager.last_alpha());
#else
	const __m128d alpha_sse = _mm_set1_pd(manager.last_alpha());
#endif

	uint32_t nb_selected = 0;

	const uint32_t* plotted = manager.get_plotted();
	const uint32_t nrows = manager.get_nrows();
	const uint32_t nrows_sse = nrows & ~31U;

	const uint32_t* buffer = use_selectable ? data.buffer_selectable().buffer() : data.buffer_selected().buffer();

	BENCH_START(b);
	PVRow i;
#pragma omp parallel for num_threads(PVCore::PVHardwareConcurrency::get_physical_core_number())
	for (i = 0; i < nrows_sse; i += 32) {
		// Compute one chunk of the selection
		int32_t chunk = 0;
		for (int j = 0; j < 32; j += 4) {
			const __m128i y_sse = _mm_load_si128((__m128i const*) &plotted[i+j]);
			const __m128i mask_y = picviz_mm_cmprange_in_epu32(y_sse, v_min_sse, v_max_sse);

			if (!_mm_test_all_zeros(mask_y, _mm_set1_epi32(0xFFFFFFFFU))) {

				// Get counters from the histogram
				const __m128i base_sse = _mm_srli_epi32(y_sse, zoom_shift);
				const __m128i p_sse = _mm_sub_epi32(base_sse, base_y_sse);

				const __m128i res_sse = picviz_mm_cmprange_epi32(p_sse, _mm_setzero_si128(), nblocks_sse);

				if (!_mm_test_all_zeros(res_sse, _mm_set1_epi32(0xFFFFFFFFU))) {
					const __m128i idx_sse = PVParallelView::PVHitGraphSSEHelpers::buffer_offset_from_y_sse(y_sse, p_sse, y_min_ref_sse, alpha_sse, zoom_mask_sse, idx_shift, zoom_shift, nbits);

					__m128i count_sse = _mm_setzero_si128();
					if (_mm_extract_epi32(res_sse, 0)) {
						count_sse = _mm_insert_epi32(count_sse, buffer[_mm_extract_epi32(idx_sse, 0)], 0);
					}
					if (_mm_extract_epi32(res_sse, 1)) {
						count_sse = _mm_insert_epi32(count_sse, buffer[_mm_extract_epi32(idx_sse, 1)], 1);
					}
					if (_mm_extract_epi32(res_sse, 2)) {
						count_sse = _mm_insert_epi32(count_sse, buffer[_mm_extract_epi32(idx_sse, 2)], 2);
					}
					if (_mm_extract_epi32(res_sse, 3)) {
						count_sse = _mm_insert_epi32(count_sse, buffer[_mm_extract_epi32(idx_sse, 3)], 3);
					}

					const __m128i mask_count = picviz_mm_cmprange_epi32(count_sse, c_min_sse, c_max_sse);

					const __m128i mask = _mm_and_si128(mask_y, mask_count);

					// Get selection bits and write them
					const int32_t sel_bits = _mm_movemask_ps(reinterpret_cast<__m128>(mask));
					chunk |= sel_bits << j;
				}
			}
		}
		sel.set_chunk32_fast_stream(Picviz::PVSelection::line_index_to_chunk32(i), chunk);
	}
	for (i = nrows_sse; i < nrows; i++) {
		const uint32_t v = plotted[i];
		if ((v >= v_min) && (v < v_max)) {
			const uint32_t c = manager.get_count_for(v);
			if ((c >= c_min) && (c < c_max)) {
				sel.set_bit_fast(i);
				++nb_selected;
				continue;
			}
		}
		sel.clear_bit_fast(i);
	}

	BENCH_END(b, "sse-invariant", nrows, sizeof(uint32_t), 1, 1);

	return nb_selected;
}

uint32_t PVParallelView::PVSelectionGenerator::compute_selection_from_plotted_range(
	const uint32_t* plotted,
	PVRow nrows,
	uint64_t y_min,
	uint64_t y_max,
	Picviz::PVSelection& sel,
	Picviz::PVSelection const& layers_sel
)
{
	return __impl::compute_selection_from_plotted_range_sse(plotted, nrows, y_min, y_max, sel, layers_sel);
}

/*****************************************************************************
 * PVParallelView::__impl::compute_selection_from_plotted_range_seq
 *****************************************************************************/
uint32_t PVParallelView::__impl::compute_selection_from_plotted_range_seq(
	const uint32_t* plotted,
	PVRow nrows,
	uint64_t y_min,
	uint64_t y_max,
	Picviz::PVSelection& sel,
	const Picviz::PVSelection& layers_sel)
{
	if (y_min == y_max) {
		return 0;
	}

	sel.ensure_allocated();

	BENCH_START(compute_selection_from_plotted_range_seq);
	for(PVRow i = 0; i < nrows; ++i) {

		const uint32_t y = plotted[i];

		if ((y >= y_min) && (y <= y_max)) {
			if (layers_sel.get_line_fast(i)) {
				sel.set_bit_fast(i);
			}
			continue;
		}

		sel.clear_bit_fast(i);
	}
	BENCH_END(compute_selection_from_plotted_range_seq, "compute_selection_from_plotted_range_seq", nrows, sizeof(uint32_t), 1, 1);

	return 0;
}

/*****************************************************************************
 * PVParallelView::__impl::compute_selection_from_plotted_range_sse
 *****************************************************************************/
uint32_t PVParallelView::__impl::compute_selection_from_plotted_range_sse(
	const uint32_t* plotted,
	PVRow nrows,
	uint64_t y_min,
	uint64_t y_max,
	Picviz::PVSelection& sel,
	const Picviz::PVSelection& layers_sel)
{
	if (y_min == y_max) {
		return 0;
	}

	sel.ensure_allocated();

	const __m128i y_min_sse = _mm_set1_epi32(y_min);
	const __m128i y_max_sse = _mm_set1_epi32(y_max);

	const uint32_t nrows_sse = nrows & ~63U;

	const __m128i sse_ff = _mm_set1_epi32(0xFFFFFFFFU);

	BENCH_START(compute_selection_from_plotted_range_sse);

#pragma omp parallel for num_threads(PVCore::PVHardwareConcurrency::get_physical_core_number()) schedule(guided, 16)
	for (PVRow i = 0; i < nrows_sse; i += 64) {
		register uint64_t chunk = 0;
		for (int j = 0; j < 64; j += 4) {
			const __m128i y_sse = _mm_load_si128((__m128i const*) &plotted[i+j]);

			const __m128i mask_y = picviz_mm_cmprange_in_epu32(y_sse, y_min_sse, y_max_sse);

			if (!(_mm_test_all_zeros(mask_y, sse_ff))) {
				const uint64_t sel_bits = _mm_movemask_ps(reinterpret_cast<__m128>(mask_y));
				chunk |= sel_bits << j;
			}
		}
		const PVRow chunk_idx = Picviz::PVSelection::line_index_to_chunk(i);
		sel.set_chunk_fast(chunk_idx, chunk & layers_sel.get_chunk_fast(chunk_idx));
	}
	for (PVRow i = nrows_sse; i < nrows; i++) {
		const uint32_t y = plotted[i];

		if ((y >= y_min) && (y <= y_max)) {
			if (layers_sel.get_line_fast(i)) {
				sel.set_bit_fast(i);
				continue;
			}
		}

		sel.clear_bit_fast(i);
	}

	BENCH_END(compute_selection_from_plotted_range_sse, "compute_selection_from_plotted_range_sse", nrows, sizeof(uint32_t), Picviz::PVSelection::line_index_to_chunk(nrows), sizeof(uint64_t));

	return 0;
}

/*****************************************************************************
 * PVParallelView::PVSelectionGenerator::compute_selection_from_plotteds_ranges
 *****************************************************************************/
uint32_t PVParallelView::PVSelectionGenerator::compute_selection_from_plotteds_ranges(
const uint32_t* y1_plotted,
	const uint32_t* y2_plotted,
	const PVRow nrows,
	const QRectF& rect,
	Picviz::PVSelection& sel,
	Picviz::PVSelection const& layers_sel
)
{
	return __impl::compute_selection_from_plotteds_ranges_sse(y1_plotted, y2_plotted, nrows, rect, sel, layers_sel);
}

/*****************************************************************************
 * PVParallelView::PVSelectionGenerator::compute_selection_from_plotted_ranges_seq
 *****************************************************************************/
uint32_t PVParallelView::__impl::compute_selection_from_plotted_ranges_seq(
	const uint32_t* y1_plotted,
	const uint32_t* y2_plotted,
	const PVRow nrows,
	const QRectF& rect,
	Picviz::PVSelection& sel,
	Picviz::PVSelection const& layers_sel
)
{
	if (rect.isNull()) {
		return 0;
	}

	sel.ensure_allocated();

	const uint32_t y1_min = PVCore::clamp(floor(rect.left()),  0.0, (double) UINT32_MAX);
	const uint32_t y1_max = PVCore::clamp(ceil(rect.right()), 0.0, (double) UINT32_MAX);

	const uint32_t y2_min = PVCore::clamp(floor(rect.top()), 0.0, (double) UINT32_MAX);
	const uint32_t y2_max = PVCore::clamp(ceil(rect.bottom()), 0.0, (double) UINT32_MAX);

	BENCH_START(scatter_view_plotted_selection_serial);
	for(PVRow i = 0; i < nrows; ++i) {

		const uint32_t y1 = y1_plotted[i];
		const uint32_t y2 = y2_plotted[i];

		if ((y1 >= y1_min) && (y1 <= y1_max) && (y2 >= y2_min) && (y2 <= y2_max)) {
			if (layers_sel.get_line_fast(i)) {
				sel.set_bit_fast(i);
			}
			continue;
		}

		sel.clear_bit_fast(i);
	}
	BENCH_END(scatter_view_plotted_selection_serial, "scatter_view_plotted_selection_serial", 2*nrows, sizeof(uint32_t), 1, 1);

	return 0;
}

/*****************************************************************************
 * PVParallelView::__impl::compute_selection_from_plotteds_ranges_sse
 *****************************************************************************/
uint32_t PVParallelView::__impl::compute_selection_from_plotteds_ranges_sse(
	const uint32_t* y1_plotted,
	const uint32_t* y2_plotted,
	const PVRow nrows,
	const QRectF& rect,
	Picviz::PVSelection& sel,
	Picviz::PVSelection const& layers_sel
)
{
	if (rect.isNull()) {
		return 0;
	}

	sel.ensure_allocated();

	const uint32_t y1_min = PVCore::clamp(floor(rect.left()), 0.0, (double) UINT32_MAX);
	const uint32_t y1_max = PVCore::clamp(ceil(rect.right()), 0.0, (double) UINT32_MAX);

	const uint32_t y2_min = PVCore::clamp(floor(rect.top()),   0.0, (double) UINT32_MAX);
	const uint32_t y2_max = PVCore::clamp(ceil(rect.bottom()), 0.0, (double) UINT32_MAX);

	const __m128i y1_min_sse = _mm_set1_epi32(y1_min);
	const __m128i y1_max_sse = _mm_set1_epi32(y1_max);

	const __m128i y2_min_sse = _mm_set1_epi32(y2_min);
	const __m128i y2_max_sse = _mm_set1_epi32(y2_max);

	const uint32_t nrows_sse = nrows & ~63U;

	const __m128i sse_ff = _mm_set1_epi32(0xFFFFFFFFU);

	BENCH_START(compute_selection_from_plotteds_ranges_sse);

#pragma omp parallel for num_threads(PVCore::PVHardwareConcurrency::get_physical_core_number()) schedule(guided, 16)
	for (PVRow i = 0; i < nrows_sse; i += 64) {
		register uint64_t chunk = 0;
		for (int j = 0; j < 64; j += 4) {
			const __m128i y1_sse = _mm_load_si128((__m128i const*) &y1_plotted[i+j]);

			const __m128i mask_y1 = picviz_mm_cmprange_in_epu32(y1_sse, y1_min_sse, y1_max_sse);

			if (!(_mm_test_all_zeros(mask_y1, sse_ff))) {
				const __m128i y2_sse = _mm_load_si128((__m128i const*) &y2_plotted[i+j]);

				const __m128i mask_y2 = picviz_mm_cmprange_in_epu32(y2_sse, y2_min_sse, y2_max_sse);
				const __m128i mask_y1_y2 = _mm_and_si128(mask_y1, mask_y2);

				const uint64_t sel_bits = _mm_movemask_ps(reinterpret_cast<__m128>(mask_y1_y2));
				chunk |= sel_bits << j;
			}
		}
		const PVRow chunk_idx = Picviz::PVSelection::line_index_to_chunk(i);
		sel.set_chunk_fast(chunk_idx, chunk & layers_sel.get_chunk_fast(chunk_idx));
	}
	for (PVRow i = nrows_sse; i < nrows; i++) {
		const uint32_t y1 = y1_plotted[i];
		const uint32_t y2 = y2_plotted[i];

		if ((y1 >= y1_min) && (y1 <= y1_max) && (y2 >= y2_min) && (y2 <= y2_max)) {
			if (layers_sel.get_line_fast(i)) {
				sel.set_bit_fast(i);
				continue;
			}
		}

		sel.clear_bit_fast(i);
	}

	BENCH_END(compute_selection_from_plotteds_ranges_sse, "compute_selection_from_plotteds_ranges_sse", 2*nrows, sizeof(uint32_t), Picviz::PVSelection::line_index_to_chunk(nrows), sizeof(uint64_t));

	return 0;
}

/*****************************************************************************
 * PVParallelView::PVSelectionGenerator::process_selection
 *****************************************************************************/
void PVParallelView::PVSelectionGenerator::process_selection(Picviz::PVView_sp view_sp, bool use_modifiers /*= true*/)
{
	PVHive::PVActor<Picviz::PVView> view_actor;
	PVHive::get().register_actor(view_sp, view_actor);

	unsigned int modifiers = (unsigned int) QApplication::keyboardModifiers();
	/* We don't care about a keypad button being pressed */
	modifiers &= ~Qt::KeypadModifier;

	/* Can't use a switch case here as Qt::ShiftModifier and Qt::ControlModifier aren't really
	 * constants */
	if (use_modifiers && modifiers == AND_MODIFIER) {
		view_actor.call<FUNC(Picviz::PVView::set_square_area_mode)>(Picviz::PVStateMachine::AREA_MODE_INTERSECT_VOLATILE);
	}
	else if (use_modifiers && modifiers == NAND_MODIFIER) {
		view_actor.call<FUNC(Picviz::PVView::set_square_area_mode)>(Picviz::PVStateMachine::AREA_MODE_SUBSTRACT_VOLATILE);
	}
	else if (use_modifiers && modifiers == OR_MODIFIER) {
		view_actor.call<FUNC(Picviz::PVView::set_square_area_mode)>(Picviz::PVStateMachine::AREA_MODE_ADD_VOLATILE);
	}
	else {
		view_actor.call<FUNC(Picviz::PVView::set_square_area_mode)>(Picviz::PVStateMachine::AREA_MODE_SET_WITH_VOLATILE);
	}

	/* Commit the previous volatile selection */
	view_actor.call<FUNC(Picviz::PVView::commit_volatile_in_floating_selection)>();

	view_actor.call<FUNC(Picviz::PVView::process_real_output_selection)>();

	PVHive::get().unregister_actor(view_actor);
}
