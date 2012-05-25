#ifndef PVSELECTIONSQUARE_H_
#define PVSELECTIONSQUARE_H_

#include <picviz/PVSelection.h>
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

	void compute_selection(PVZoneID zid, QRect rect)
	{
		Picviz::PVSelection* sel = new Picviz::PVSelection();
		uint32_t width = _zm.get_zone_width(zid);

		PVZoneTree& ztree = _zm.get_zone_tree<PVZoneTree>(zid);
		PVZoneTree::PVBranch* treeb = ztree.get_treeb();
		PVParallelView::PVBCode code_b;

		PVLineEqInt line;
		line.b = -width;
		for (uint32_t b = 0 ; b < NBUCKETS; b++)
		{
			PVRow r =  ztree.get_first_elt_of_branch(b);
			if(r == PVROW_INVALID_VALUE) {
				continue;
			}
			code_b.int_v = b;
			uint32_t y1 = code_b.s.l;
			uint32_t y2 = code_b.s.r;

			line.a = y2 - y1;
			line.c = y1;

			bool a = line(rect.topLeft().x(), rect.topLeft().y()) >= 0;
			bool b = line(rect.topRight().x(), rect.topRight().y()) >=0;
			bool c = line(rect.bottomLeft().x(), rect.bottomLeft().y()) >=0;
			bool d = line(rect.bottomRight().x(), rect.bottomRight().y()) >=0;

			bool is_line_selected = ((a | b | c | d) & (!(a & b & c & d)));

			if (is_line_selected)
			{
				for (size_t i = 0; i < treeb[b].count; i++) {
					sel->set_bit_fast(treeb[b].p[i]);
				}
				ztree._sel_elts[b] = r;
			}
		}
	}

	PVZonesManager& _zm;
};

}

#endif /* PVSELECTIONSQUARE_H_ */
