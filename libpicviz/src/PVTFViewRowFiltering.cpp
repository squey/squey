#include <picviz/PVTFViewRowFiltering.h>
#include <picviz/PVSelRowFilteringFunction.h>
#include <picviz/PVView.h>

Picviz::PVSelection Picviz::PVTFViewRowFiltering::operator()(PVView const& view_src, PVView const& view_dst, PVSelection const& sel_org) const
{
	// AG: the algorithm here is hard coded, and the idea in a close future is to have the user being able
	// to create this kind of algorithm with a diagram... !
	PVSelection sel_ret;
	sel_ret.select_none();

	// For each line of sel_org, create a selection that goes with view_dst
	// Then, merge this selection into the final one.
	PVRow nlines_sel = view_src.get_row_count();
	PVSelection sel_line;
	for (PVRow r = 0; r < nlines_sel; r++) {
		if (!sel_org.get_line(r)) {
			continue;
		}

		sel_line.select_all();
		// sel_line will be filtered by the different RFF (row filtering functions).
		foreach(PVSelRowFilteringFunction_p const& rff_p, _rffs) {
			(*rff_p)(r, view_src, view_dst, sel_line);
		}

		sel_ret |= sel_line;
	}

	return sel_ret;
}
