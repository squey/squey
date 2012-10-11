/**
 * \file PVQuadTree.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvparallelview/PVQuadTree.h>

/*****************************************************************************
 * PVParallelView::__impl::f_get_first
 *****************************************************************************/

size_t PVParallelView::__impl::f_get_first(const PVQuadTreeEntry &e,
                                           uint32_t,
                                           uint32_t, uint32_t,
                                           const PVCore::PVHSVColor *,
                                           PVQuadTreeEntry *entries)
{
	*entries = e;
	return 1;
}

/*****************************************************************************
 * PVParallelView::__impl::f_get_entry_sel
 *****************************************************************************/

void PVParallelView::__impl::f_get_entry_sel(const pvquadtree_entries_t &entries,
                                             const Picviz::PVSelection &selection,
                                             const PVCore::PVHSVColor* /*colors*/,
                                             pvquadtree_entries_t &result)
{
	for(unsigned i = 0; i < entries.size(); ++i) {
		const PVQuadTreeEntry &e = entries.at(i);
		if(selection.get_line(e.idx)) {
			result.push_back(e);
		}
	}
}
