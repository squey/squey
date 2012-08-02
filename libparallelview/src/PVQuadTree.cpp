/**
 * \file PVQuadTree.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvparallelview/PVQuadTree.h>

/*****************************************************************************
 * PVParallelView::__impl::f_get_first_bci
 *****************************************************************************/

size_t PVParallelView::__impl::f_get_first_bci(const PVQuadTreeEntry &e,
                                               uint32_t y_start,
                                               uint32_t shift, uint32_t mask,
                                               const PVHSVColor *colors,
                                               PVBCICode *code)
{
	code->s.idx = e.idx;
	code->s.l = ((e.y1 - y_start) >> shift) & mask;
	code->s.r = ((e.y2 - y_start) >> shift) & mask;
	code->s.color = colors[e.idx].h();
	return 1;
}

/*****************************************************************************
 * PVParallelView::__impl::f_get_first_bci_sel
 *****************************************************************************/

size_t PVParallelView::__impl::f_get_first_bci_sel(const pvquadtree_entries_t &entries,
                                                   const Picviz::PVSelection &selection,
                                                   const PVHSVColor *colors,
                                                   PVBCICode *code)
{
	for(unsigned i = 0; i < entries.size(); ++i) {
		const PVQuadTreeEntry &e = entries.at(i);
		if(selection.get_line(e.idx)) {
			code->s.idx = e.idx;
			code->s.l = e.y1 >> (32 - NBITS_INDEX);
			code->s.r = e.y2 >> (32 - NBITS_INDEX);
			code->s.color = colors[e.idx].h();
			return 1;
		}
	}
	return 0;
}

/*****************************************************************************
 * PVParallelView::__impl::f_get_entry_sel
 *****************************************************************************/

void PVParallelView::__impl::f_get_entry_sel(const pvquadtree_entries_t &entries,
                                             const Picviz::PVSelection &selection,
                                             const PVHSVColor */*colors*/,
                                             pvquadtree_entries_t &result)
{
	for(unsigned i = 0; i < entries.size(); ++i) {
		const PVQuadTreeEntry &e = entries.at(i);
		if(selection.get_line(e.idx)) {
			result.push_back(e);
		}
	}
}

/*****************************************************************************
 * PVParallelView::__impl::f_get_bci_sel
 *****************************************************************************/

void PVParallelView::__impl::f_get_bci_sel(const pvquadtree_entries_t &entries,
                                           const Picviz::PVSelection &selection,
                                           const PVHSVColor *colors,
                                           pvquadtree_bcicodes_t &result)
{
	for(unsigned i = 0; i < entries.size(); ++i) {
		const PVQuadTreeEntry &e = entries.at(i);
		if(selection.get_line(e.idx)) {
			PVParallelView::PVBCICode code;
			code.s.idx = e.idx;
			code.s.l = e.y1 >> (32 - NBITS_INDEX);
			code.s.r = e.y2 >> (32 - NBITS_INDEX);
			code.s.color = colors[e.idx].h();
			result.push_back(code);
		}
	}
}
