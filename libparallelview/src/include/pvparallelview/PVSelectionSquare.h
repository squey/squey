/**
 * \file PVSelectionSquare.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVSELECTIONSQUARE_H_
#define PVSELECTIONSQUARE_H_

#include <picviz/PVSelection.h>
#include <pvparallelview/common.h>
#include <pvparallelview/PVZoneTree.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVBCode.h>

namespace PVParallelView
{

struct PVLineEqInt
{
	int a;
	int b;
	int c;
	inline int operator()(int X, int Y) const { return a*X+b*Y+c; }
};

class PVSelectionSquare
{
public:
	PVSelectionSquare(PVZonesManager& zm) : _zm(zm) {};

	void compute_selection(PVZoneID zid, QRect rect, Picviz::PVSelection& sel)
	{
		sel.select_none();
		int32_t width = _zm.get_zone_width(zid);

		PVZoneTree& ztree = _zm.get_zone_tree<PVZoneTree>(zid);
		PVZoneTree::PVBranch* treeb = ztree.get_treeb();
		PVParallelView::PVBCode code_b;

		if (rect.isNull()) {
			memset(ztree._sel_elts, PVROW_INVALID_VALUE, NBUCKETS*sizeof(PVRow));
			return;
		}

		PVLineEqInt line;
		line.b = -width;
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
				for (size_t i = 0; i < treeb[branch].count; i++) {
					sel.set_bit_fast(treeb[branch].p[i]);
				}
				ztree._sel_elts[branch] = r;
			}
			else {
				ztree._sel_elts[branch] = PVROW_INVALID_VALUE;
			}
		}
	}

	PVZonesManager& _zm;
};

}

#endif /* PVSELECTIONSQUARE_H_ */
