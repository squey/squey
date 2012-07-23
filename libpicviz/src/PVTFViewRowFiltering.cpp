/**
 * \file PVTFViewRowFiltering.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/picviz_bench.h>

#include <picviz/PVTFViewRowFiltering.h>
#include <picviz/PVSelRowFilteringFunction.h>
#include <picviz/PVView.h>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>

void Picviz::PVTFViewRowFiltering::pre_process(PVView const& view_src, PVView const& view_dst)
{
	foreach(PVSelRowFilteringFunction_p const& rff_p, _rffs) {
		rff_p->pre_process(view_src, view_dst);
	}
}

bool Picviz::PVTFViewRowFiltering::all_rff_or_operation() const
{
	if (_rffs.size() <= 1) {
		return true;
	}
	list_rff_t::const_iterator it = _rffs.begin(); it++;
	for (; it != _rffs.end(); it++) {
		if ((*it)->get_combination_op() != PVCore::PVBinaryOperation::OR) {
			return false;
		}
	}
	return true;
}

Picviz::PVSelection Picviz::PVTFViewRowFiltering::operator()(PVView const& view_src, PVView const& view_dst, PVSelection const& sel_org) const
{
	if (_rffs.size() == 0) {
		return std::move(PVSelection());
	}

	// AG: the algorithm here is hard coded, and the idea in a close future is to have the user being able
	// to create this kind of algorithm with a diagram... !
	BENCH_START(merge);


	const PVRow nlines_sel = view_src.get_row_count();
	// Special case when all RFF have "OR" operation. We can be really faster by always writing into the same
	// selection !
	tbb::enumerable_thread_specific<PVSelection> tls_sel;

	if (all_rff_or_operation()) {
		PVLOG_INFO("Correlation: only OR operations, optimizing process...\n");
		double time_for = 0.0;
#pragma omp parallel num_threads(12) reduction(+:time_for)
		{
#pragma omp single
			sel_org.visit_selected_lines([&](PVRow r)
				{
			#pragma omp task default(shared)
					{
						Picviz::PVSelection& task_sel = tls_sel.local();
						foreach(PVSelRowFilteringFunction_p const& rff_p, _rffs) {
							(*rff_p)(r, view_src, view_dst, task_sel);
						}
					}
				},
				nlines_sel);
#pragma omp taskwait
		}

		if (tls_sel.size() == 0) {
			return std::move(PVSelection());
		}
		PVSelection& final_sel = *tls_sel.begin();
		// Merge all TLS selections
		tbb::enumerable_thread_specific<PVSelection>::const_iterator it_sel = tls_sel.begin();
		it_sel++;
		BENCH_START(sel_red);
		for (; it_sel != tls_sel.end(); it_sel++) {
			final_sel.or_optimized(*it_sel);
		}
		BENCH_END(sel_red, "selection reduction", 1, 1, 1, 1);
		BENCH_END(merge, "merge", 1, 1, 1, 1);
		return std::move(final_sel);
	}
		
	PVSelection sel_ret;

	PVLOG_DEBUG("PVTFViewRowFiltering::operator()) has %u RFF:\n", _rffs.size());
	foreach(PVSelRowFilteringFunction_p const& rff_p, _rffs) {
		PVCore::dump_argument_list(rff_p->get_args());
	}

	// For each line of sel_org, create a selection that goes with view_dst
	// Then, merge this selection into the final one.
	PVSelection sel_tmp_row;
	PVSelection sel_tmp_rff;

	for (PVRow r = 0; r < nlines_sel; r++) {
		if (!sel_org.get_line(r)) {
			continue;
		}

		sel_tmp_row.select_none();
		foreach(PVSelRowFilteringFunction_p const& rff_p, _rffs) {

			sel_tmp_rff.select_none();
			(*rff_p)(r, view_src, view_dst, sel_tmp_rff);

			switch (rff_p->get_combination_op() /*index*/) {
				case PVCore::PVBinaryOperation::OR:
				{
					//sel_tmp_row |= sel_tmp_rff;
					sel_tmp_row.or_optimized(sel_tmp_rff);
					break;
				}
				case PVCore::PVBinaryOperation::AND:
				{
					//sel_tmp_row &= sel_tmp_rff;
					sel_tmp_row.and_optimized(sel_tmp_rff);
					break;
				}
				case PVCore::PVBinaryOperation::XOR:
				{
					sel_tmp_row ^= sel_tmp_rff;
					break;
				}
				case PVCore::PVBinaryOperation::OR_NOT:
				{
					sel_tmp_row.or_not(sel_tmp_rff);
					break;
				}
				case PVCore::PVBinaryOperation::AND_NOT:
				{
					sel_tmp_row.and_not(sel_tmp_rff);
					break;
				}
				case PVCore::PVBinaryOperation::XOR_NOT:
				{
					sel_tmp_row.xor_not(sel_tmp_rff);
					break;
				}
				default:
				{
					assert(false);
				}
			}
		}

		sel_ret.or_optimized(sel_tmp_row);
	}

	BENCH_END(merge, "merge", 1, 1, 1, 1);

	return sel_ret;
}


void Picviz::PVTFViewRowFiltering::to_xml(QDomElement& elt) const
{
	list_rff_t::const_iterator it;
	for (it = _rffs.begin(); it != _rffs.end(); it++) {
		QDomElement rff_elt = elt.ownerDocument().createElement("rff");
		(*it)->to_xml(rff_elt);
		elt.appendChild(rff_elt);
	}
}

void Picviz::PVTFViewRowFiltering::from_xml(QDomElement const& elt)
{
	_rffs.clear();
	QDomElement child = elt.firstChildElement("rff");
	for (; !child.isNull(); child = child.nextSiblingElement("rff")) {
		PVSelRowFilteringFunction_p rff = PVSelRowFilteringFunction::from_xml(child);
		if (rff) {
			push_rff(rff);
		}
	}
}
