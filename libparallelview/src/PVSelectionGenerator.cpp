/**
 * \file PVSelectionGenerator.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvparallelview/PVSelectionGenerator.h>
#include <pvparallelview/PVBCode.h>
#include <pvkernel/core/picviz_bench.h>
#include <picviz/PVSelection.h>
#include <pvparallelview/PVZoneTree.h>
#include <pvparallelview/PVZonesManager.h>

uint32_t PVParallelView::PVSelectionGenerator::compute_selection_from_rect(PVZoneID zid, QRect rect, Picviz::PVSelection& sel)
{
	uint32_t nb_selected = 0;

	sel.select_none();
	int32_t width = _zm.get_zone_width(zid);

	PVZoneTree& ztree = _zm.get_zone_tree<PVZoneTree>(zid);
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
	Picviz::PVSelection& sel
)
{
	uint32_t nb_selected = 0;

	sel.select_none();

	PVParallelView::PVBCode code_b;

	if (zid < _zm.get_number_zones()) {
		PVZoneTree& ztree = _zm.get_zone_tree<PVZoneTree>(zid);

		for (uint32_t branch = 0 ; branch < NBUCKETS; branch++)	{
			PVRow r =  ztree.get_first_elt_of_branch(branch);
			if(r == PVROW_INVALID_VALUE) {
				ztree._sel_elts[branch] = PVROW_INVALID_VALUE;
				continue;
			}

			code_b.int_v = branch;
			uint32_t y1 = code_b.s.l;

			bool is_line_selected = false;
			for (auto range : ranges) {
				if (y1 >= range.first && y1 <= range.second) {
					is_line_selected = true;
					break;
				}
			}

			if(is_line_selected) {
				ztree._sel_elts[branch] = r;
				uint32_t branch_count = ztree.get_branch_count(branch);
				for (size_t i = 0; i < ztree.get_branch_count(branch); i++) {
					sel.set_bit_fast(ztree.get_branch_element(branch, i));
				}
				nb_selected += branch_count;
			}
			else {
				ztree._sel_elts[branch] = PVROW_INVALID_VALUE;
			}
		}
	} else {
		// for the last zid, we must process bci.s.r in the last zone tree
		PVZoneTree& ztree = _zm.get_zone_tree<PVZoneTree>(zid - 1);

		for (uint32_t branch = 0 ; branch < NBUCKETS; branch++)	{
			PVRow r =  ztree.get_first_elt_of_branch(branch);
			if(r == PVROW_INVALID_VALUE) {
				ztree._sel_elts[branch] = PVROW_INVALID_VALUE;
				continue;
			}

			code_b.int_v = branch;
			uint32_t y1 = code_b.s.r;

			bool is_line_selected = false;
			for (auto range : ranges) {
				if (y1 >= range.first && y1 <= range.second) {
					is_line_selected = true;
					break;
				}
			}

			if(is_line_selected) {
				ztree._sel_elts[branch] = r;
				uint32_t branch_count = ztree.get_branch_count(branch);
				for (size_t i = 0; i < ztree.get_branch_count(branch); i++) {
					sel.set_bit_fast(ztree.get_branch_element(branch, i));
				}
				nb_selected += branch_count;
			}
			else {
				ztree._sel_elts[branch] = PVROW_INVALID_VALUE;
			}
		}
	}

	return nb_selected;
}

