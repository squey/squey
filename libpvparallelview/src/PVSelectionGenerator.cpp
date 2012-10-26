/**
 * \file PVSelectionGenerator.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/picviz_bench.h>

#include <picviz/PVSelection.h>

#include <pvparallelview/PVSelectionGenerator.h>
#include <pvparallelview/PVBCode.h>
#include <pvparallelview/PVZoneTree.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVAbstractAxisSlider.h>

uint32_t PVParallelView::PVSelectionGenerator::compute_selection_from_rect(PVZoneID zid, QRect rect, Picviz::PVSelection& sel)
{
	uint32_t nb_selected = 0;

	int32_t width = _lines_view.get_zone_width(zid);

	PVZoneTree& ztree = get_zones_manager().get_zone_tree<PVZoneTree>(zid);
	PVParallelView::PVBCode code_b;

	if (rect.isNull()) {
		memset(ztree._sel_elts, PVROW_INVALID_VALUE, NBUCKETS*sizeof(PVRow));
		return 0;
	}

	BENCH_START(compute_selection_from_rect);
	PVLineEqInt line;
	line.b = -width;
//#pragma omp parallel for num_threads(4) reduction(+:nb_selected)
	for (uint32_t branch = 0 ; branch < NBUCKETS; branch++)
	{
		PVRow r =  ztree.get_first_elt_of_branch(branch);
		if(r == PVROW_INVALID_VALUE) {
			ztree._sel_elts[branch] = PVROW_INVALID_VALUE;
			continue;
		}
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
			ztree._sel_elts[branch] = r;
			uint32_t branch_count = ztree.get_branch_count(branch);
			for (size_t i = 0; i < branch_count; i++) {
				const PVRow cur_r = ztree.get_branch_element(branch, i);
				//sel.set_bit_fast(cur_r);
//#pragma omp atomic
				sel.get_buffer()[Picviz::PVSelection::line_index_to_chunk(cur_r)] |= 1UL<<Picviz::PVSelection::line_index_to_chunk_bit(cur_r);
			}
			nb_selected += branch_count;
		}
		else {
			ztree._sel_elts[branch] = PVROW_INVALID_VALUE;
		}
	}
	BENCH_END(compute_selection_from_rect, "compute_selection", sizeof(PVRow), NBUCKETS, 1, 1);

	return nb_selected;
}

uint32_t PVParallelView::PVSelectionGenerator::compute_selection_from_sliders(
	PVZoneID zid,
	const typename PVAxisGraphicsItem::selection_ranges_t& ranges,
	Picviz::PVSelection& sel)
{
	uint32_t nb_selected = 0;

	sel.select_none();

	PVParallelView::PVBCode code_b;

	if (zid < get_zones_manager().get_number_zones()) {
		// process the left side of zones
		PVZoneTree& ztree = get_zones_manager().get_zone_tree<PVZoneTree>(zid);

		memset(ztree._sel_elts, 0xFFFFFFFF, NBUCKETS * sizeof(ztree._sel_elts[0]));

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

					ztree._sel_elts[branch] = r;
					uint32_t branch_count = ztree.get_branch_count(branch);
					for (size_t i = 0; i < branch_count; ++i) {
						sel.set_bit_fast(ztree.get_branch_element(branch, i));
					}
					nb_selected += branch_count;
				}
			}

			if (need_zzt_min) {
				PVZoomedZoneTree& zztree = get_zones_manager().get_zone_tree<PVZoomedZoneTree>(zid);
				nb_selected += zztree.compute_selection_y1(zzt_min_idx, range_min,
				                                           range_max, sel);
			}

			if (need_zzt_max) {
				PVZoomedZoneTree& zztree = get_zones_manager().get_zone_tree<PVZoomedZoneTree>(zid);
				nb_selected += zztree.compute_selection_y1(zzt_max_idx, range_min,
				                                           range_max, sel);
			}
		}
	} else {
		// process the right side of zones
		PVZoneTree& ztree = get_zones_manager().get_zone_tree<PVZoneTree>(zid - 1);

		memset(ztree._sel_elts, 0xFFFFFFFF, NBUCKETS * sizeof(ztree._sel_elts[0]));

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

					ztree._sel_elts[branch] = r;
					uint32_t branch_count = ztree.get_branch_count(branch);
					for (size_t i = 0; i < branch_count; ++i) {
						sel.set_bit_fast(ztree.get_branch_element(branch, i));
					}
					nb_selected += branch_count;
				}
			}

			if (need_zzt_min) {
				PVZoomedZoneTree& zztree = get_zones_manager().get_zone_tree<PVZoomedZoneTree>(zid - 1);
				nb_selected += zztree.compute_selection_y2(zzt_min_idx, range_min,
				                                           range_max, sel);
			}

			if (need_zzt_max) {
				PVZoomedZoneTree& zztree = get_zones_manager().get_zone_tree<PVZoomedZoneTree>(zid - 1);
				nb_selected += zztree.compute_selection_y2(zzt_max_idx, range_min,
				                                           range_max, sel);
			}
		}
	}

	return nb_selected;
}

