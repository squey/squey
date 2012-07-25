/**
 * \file PVRFFAxesBind.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/picviz_bench.h>
#include "PVRFFAxesBind.h"
#include <picviz/PVSparseSelection.h>
#include <picviz/PVView.h>

#include <pvkernel/core/PVAxisIndexType.h>

#include <omp.h>

Picviz::PVRFFAxesBind::PVRFFAxesBind(PVCore::PVArgumentList const& l)
{
	INIT_FILTER(PVRFFAxesBind, l);
}

DEFAULT_ARGS_FUNC(Picviz::PVRFFAxesBind)
{
	PVCore::PVArgumentList args;
	args[PVCore::PVArgumentKey("axis_org", "Axis of original view")].setValue(PVCore::PVAxisIndexType(0));
	args[PVCore::PVArgumentKey("axis_dst", "Axis of final view")].setValue(PVCore::PVAxisIndexType(0));
	return args;
}

void Picviz::PVRFFAxesBind::set_args(PVCore::PVArgumentList const& args)
{
	PVSelRowFilteringFunction::set_args(args);
	_axis_org = args["axis_org"].value<PVCore::PVAxisIndexType>().get_original_index();
	_axis_dst = args["axis_dst"].value<PVCore::PVAxisIndexType>().get_original_index();
}

QString Picviz::PVRFFAxesBind::get_human_name_with_args(const PVView& src_view, const PVView& dst_view) const
{
	return get_human_name() + " (" + src_view.get_original_axis_name(_axis_org) + " -> " + dst_view.get_original_axis_name(_axis_dst) + ")";
}

void Picviz::PVRFFAxesBind::do_pre_process(PVView const& /*view_org*/, PVView const& view_dst)
{
	BENCH_START(b);
	PVRow nrows = view_dst.get_row_count();
	const PVMapped* m_dst = view_dst.get_parent<PVMapped>();
//#pragma omp parallel num_threads(NTHREADS)
//	{
		//hash_rows &dst_values(_dst_values[omp_get_thread_num()]);
		hash_rows &dst_values(_dst_values);
		dst_values.clear();

//#pragma omp for
		for (PVRow r = 0; r < nrows; r++) {
			//dst_values[m_dst->get_value(r, _axis_dst)].set(r);
			dst_values[m_dst->get_value(r, _axis_dst)].push_back(r);
		}
	//}
	BENCH_END(b, "preprocess", 1, 1, 1, 1);
}

void Picviz::PVRFFAxesBind::operator()(PVRow row_org, PVView const& view_org, PVView const& /*view_dst*/, PVSparseSelection& sel_dst) const
{
	const PVMapped* m_org = view_org.get_mapped_parent();
	float mf_org = m_org->get_value(row_org, _axis_org);

	hash_rows const& dst_values(_dst_values);
	hash_rows::const_iterator it_f = dst_values.find(mf_org);
	if (it_f == dst_values.end()) {
		return;
	}

	/*Picviz::PVSparseSelection const& ssel(it_f->second);
	sel_dst = ssel;*/

	std::vector<PVRow> const& rows(it_f->second);
	for (PVRow r: rows) {
		sel_dst.set(r);
	}
}

void Picviz::PVRFFAxesBind::process_or(PVRow row_org, PVView const& view_org, PVView const& /*view_dst*/, PVSelection& sel_dst) const
{
	const PVMapped* m_org = view_org.get_mapped_parent();
	float mf_org = m_org->get_value(row_org, _axis_org);

	hash_rows const& dst_values(_dst_values);
	hash_rows::const_iterator it_f = dst_values.find(mf_org);
	if (it_f == dst_values.end()) {
		return;
	}

	/*Picviz::PVSparseSelection const& ssel(it_f->second);
	sel_dst |= ssel;*/
	std::vector<PVRow> const& rows(it_f->second);
	for (PVRow r: rows) {
		sel_dst.set_bit_fast(r);
	}
}
