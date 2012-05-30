#include <picviz/PVTFViewRowFiltering.h>
#include <picviz/PVSelRowFilteringFunction.h>
#include <picviz/PVView.h>

#include <pvkernel/core/picviz_bench.h>

void Picviz::PVTFViewRowFiltering::pre_process(PVView const& view_src, PVView const& view_dst)
{
	foreach(PVSelRowFilteringFunction_p const& rff_p, _rffs) {
		rff_p->pre_process(view_src, view_dst);
	}
}

Picviz::PVSelection Picviz::PVTFViewRowFiltering::operator()(PVView const& view_src, PVView const& view_dst, PVSelection const& sel_org) const
{
	// AG: the algorithm here is hard coded, and the idea in a close future is to have the user being able
	// to create this kind of algorithm with a diagram... !
	BENCH_START(merge);

	PVSelection sel_ret;
	sel_ret.select_none();

	// For each line of sel_org, create a selection that goes with view_dst
	// Then, merge this selection into the final one.
	PVRow nlines_sel = view_src.get_row_count();
	PVSelection sel_line;

	PVLOG_INFO("PVTFViewRowFiltering::operator()) has %u RFF:\n", _rffs.size());
	foreach(PVSelRowFilteringFunction_p const& rff_p, _rffs) {
		PVCore::dump_argument_list(rff_p->get_args());
	}

	PVSelection sel_tmp_row;
	PVSelection sel_tmp_rff;
//#pragma omp parallel for
	for (PVRow r = 0; r < nlines_sel; r++) {
		if (!sel_org.get_line(r)) {
			continue;
		}

		//sel_line.select_none();
		// sel_line will be filtered by the different RFF (row filtering functions).

		int index = 0;
		sel_tmp_row.select_none();
		foreach(PVSelRowFilteringFunction_p const& rff_p, _rffs) {

			sel_tmp_rff.select_none();
			(*rff_p)(r, view_src, view_dst, sel_tmp_rff);

			switch (rff_p->get_combination_op() /*index*/) {
				case PVCore::PVBinaryOperation::OR:
				{
					sel_tmp_row |= sel_tmp_rff;
					//sel_tmp_row.or_optimized(sel_tmp_rff);
					break;
				}
				case PVCore::PVBinaryOperation::AND:
				{
					sel_tmp_row &= sel_tmp_rff;
					break;
				}
				case PVCore::PVBinaryOperation::XOR:
				{
					sel_tmp_row ^= sel_tmp_rff;
				}
				case PVCore::PVBinaryOperation::NOT:
				{
					sel_tmp_row = ~sel_tmp_rff;
					break;
				}
				default:
				{
					assert(false);
				}
			}
			index++;
		}

		sel_ret |= sel_tmp_row;
		//sel_ret.or_optimized(sel_tmp_row);
	}

	BENCH_END(merge, "merge", 1, 1, 1, 1);

	return sel_ret;
}
