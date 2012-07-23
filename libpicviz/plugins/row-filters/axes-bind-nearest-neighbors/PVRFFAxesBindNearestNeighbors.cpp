/**
 * \file PVRFFAxesBindNearestNeighbors.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/picviz_bench.h>
#include "PVRFFAxesBindNearestNeighbors.h"
#include <picviz/PVView.h>

#include <pvkernel/core/PVAxisIndexType.h>

#include <omp.h>

Picviz::PVRFFAxesBindNearestNeighbors::PVRFFAxesBindNearestNeighbors(PVCore::PVArgumentList const& l)
{
	INIT_FILTER(PVRFFAxesBindNearestNeighbors, l);
}

DEFAULT_ARGS_FUNC(Picviz::PVRFFAxesBindNearestNeighbors)
{
	PVCore::PVArgumentList args;
	args[PVCore::PVArgumentKey("axis_org", "Axis of original view")].setValue(PVCore::PVAxisIndexType(0));
	args[PVCore::PVArgumentKey("axis_dst", "Axis of final view")].setValue(PVCore::PVAxisIndexType(0));
	args[PVCore::PVArgumentKey("distance", "Maximal distance to original selection")] = 10000.0;
	return args;
}

void Picviz::PVRFFAxesBindNearestNeighbors::set_args(PVCore::PVArgumentList const& args)
{
	PVSelRowFilteringFunction::set_args(args);
	_axis_org = args["axis_org"].value<PVCore::PVAxisIndexType>().get_original_index();
	_axis_dst = args["axis_dst"].value<PVCore::PVAxisIndexType>().get_original_index();
	_distance = args["distance"].toFloat();
}

QString Picviz::PVRFFAxesBindNearestNeighbors::get_human_name_with_args(const PVView& src_view, const PVView& dst_view) const
{
	QString dist;
	dist.setNum(_distance, 'f', 2);
	return get_human_name() + " (" + src_view.get_axis_name(_axis_org) + " -> " + dst_view.get_axis_name(_axis_dst) + " [" + dist +"])";
}

void Picviz::PVRFFAxesBindNearestNeighbors::do_pre_process(PVView const& /*view_org*/, PVView const& view_dst)
{
	BENCH_START(b);

	PVRow nrows = view_dst.get_row_count();
	const PVMapped* m_dst = view_dst.get_parent<PVMapped>();

	map_rows &dst_values(_dst_values);
	dst_values.clear();

	for (PVRow r = 0; r < nrows; r++) {
		dst_values[m_dst->get_value(r, _axis_dst)].push_back(r);
	}

	BENCH_END(b, "preprocess", 1, 1, 1, 1);
}

void Picviz::PVRFFAxesBindNearestNeighbors::operator()(PVRow row_org, PVView const& view_org, PVView const& /*view_dst*/, PVSelection& sel_dst) const
{
	uint32_t* sel_buf = sel_dst.get_buffer();

	const PVMapped* m_org = view_org.get_parent<PVMapped>();
	float mf_org = m_org->get_value(row_org, _axis_org);

	map_rows const& dst_values(_dst_values);

	map_rows::const_iterator it_min = dst_values.lower_bound (mf_org - _distance);
	map_rows::const_iterator it_max = dst_values.upper_bound (mf_org + _distance);

	// rejecting out of bounds results
	if (it_min == dst_values.end() || it_max == dst_values.begin()) {
		return;
	}

	for (map_rows::const_iterator it = it_min; it != it_max; ++it) {
		std::vector<PVRow> const& rows = it->second;
		for (size_t i = 0; i < rows.size(); i++) {
			const PVRow r = rows[i];
#pragma omp atomic
			sel_buf[r>>5] |= 1U<<(r&31);
		}
	}
}
