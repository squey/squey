/**
 * \file PVLayerFilterHeatline.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <pvbase/qhashes.h>

#include "PVLayerFilterHeatline.h"

#include <pvkernel/core/picviz_bench.h>
#include <pvkernel/core/PVColor.h>
#include <pvkernel/core/PVAxisIndexType.h>
#include <pvkernel/core/PVColorGradientDualSliderType.h>
#include <pvkernel/core/PVEnumType.h>
#include <pvkernel/rush/PVUtils.h>

#include <picviz/PVView.h>

#include <tbb/concurrent_hash_map.h>
#include <tbb/enumerable_thread_specific.h>

#include <math.h>
#include <unordered_map>

#include <omp.h>

#define ARG_NAME_AXES "axes"
#define ARG_DESC_AXES "Axes"
#define ARG_NAME_SCALE "scale"
#define ARG_DESC_SCALE "Scale"
#define ARG_NAME_COLORS "colors"
#define ARG_DESC_COLORS "Colors"

/******************************************************************************
 *
 * Picviz::PVLayerFilterHeatlineBase::PVLayerFilterHeatlineBase
 *
 *****************************************************************************/
Picviz::PVLayerFilterHeatlineBase::PVLayerFilterHeatlineBase(PVCore::PVArgumentList const& l)
	: PVLayerFilter(l)
{
	INIT_FILTER(PVLayerFilterHeatlineBase, l);
}

/******************************************************************************
 *
 * DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterHeatlineBase)
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterHeatlineBase)
{
	PVCore::PVArgumentList args;
	args[PVCore::PVArgumentKey(ARG_NAME_AXES, ARG_DESC_AXES)].setValue(PVCore::PVAxisIndexType());

	PVCore::PVEnumType scale(QStringList() << "Linear" << "Log", 0);
	args[PVCore::PVArgumentKey(ARG_NAME_SCALE, ARG_DESC_SCALE)].setValue(scale);

	return args;
}

/******************************************************************************
 *
 * Picviz::PVLayerFilterHeatlineBase::get_default_args_for_view
 *
 *****************************************************************************/
PVCore::PVArgumentList Picviz::PVLayerFilterHeatlineBase::get_default_args_for_view(PVView const& view)
{
	PVCore::PVArgumentList args = get_default_args();
	// Default args with the "key" tag
	args[ARG_NAME_AXES].setValue(PVCore::PVAxisIndexType(0));
	return args;
}

/******************************************************************************
 *
 * Picviz::PVLayerFilterHeatlineBase::operator()
 *
 *****************************************************************************/
void Picviz::PVLayerFilterHeatlineBase::operator()(PVLayer& in, PVLayer &out)
{	
	BENCH_START(heatline);

	PVRush::PVNraw const& nraw = _view->get_rushnraw_parent();
	
	PVCore::PVAxisIndexType axis = _args.value(ARG_NAME_AXES).value<PVCore::PVAxisIndexType>();
	const PVCol axis_id = axis.get_original_index();
	/*if (axes.size() == 0) {
		_args = get_default_args_for_view(*_view);
		axes = _args.value(ARG_NAME_AXES).value<PVCore::PVAxesIndexType>();
		if (axes.size() == 0) {
			PVLOG_ERROR("(PVLayerFilterHeatlineBase) no key axes defined in the format and no axes selected !\n");
			if (&in != &out) {
				out = in;
			}
			return;
		}
	}*/

	bool bLog = _args.value(ARG_NAME_SCALE).value<PVCore::PVEnumType>().get_sel().compare("Log") == 0;

	out.get_selection() = in.get_selection();

	// Per-thread frequencies
	typedef std::unordered_map<std::string_tbb, PVRow> lines_hash_t;
	lines_hash_t freqs;

	const PVRow nrows = _view->get_row_count();
	freqs.reserve(nrows);

	std::vector<PVRow const*> row_values;
	row_values.resize(nrows, nullptr);

	nraw.visit_column_sel(axis_id,
		[&](const PVRow r, const char* buf, size_t size)
		{
			std::string_tbb tmp_str(buf, size);
			lines_hash_t::iterator it = freqs.find(tmp_str);
			if (it == freqs.end()) {
				it = freqs.emplace(std::move(tmp_str), 0).first;
			}
			else {
				it->second++;
			}
			row_values[r] = &it->second;
		},
		_view->get_pre_filter_layer().get_selection());

	lines_hash_t::const_iterator it;
	PVRow max_n = 0;
	for (it = freqs.begin(); it != freqs.end(); it++) {
		const PVRow cur_n = it->second;
		if (cur_n > max_n) {
			max_n = cur_n;
		}
	}

	const float max_n_log = logf(max_n);

	_view->get_pre_filter_layer().get_selection().visit_selected_lines(
		[&](const PVRow r)
		{
			assert(r < row_values.size());
			const PVRow freq = *row_values[r];
			float ratio;
			if (bLog) {
				ratio = logf(freq)/max_n_log;
			}
			else {
				ratio = (float)((double)freq/(double)max_n);
			}
			this->post(in, out, ratio, r);
		}, nrows);

	BENCH_END(heatline, "heatline", 1, 1, sizeof(PVRow), nrows);
}

QList<PVCore::PVArgumentKey> Picviz::PVLayerFilterHeatlineBase::get_args_keys_for_preset() const
{
	QList<PVCore::PVArgumentKey> keys = get_default_args().keys();
	keys.removeAll(ARG_NAME_AXES);
	return keys;
}

void Picviz::PVLayerFilterHeatlineBase::post(PVLayer& /*in*/, PVLayer& /*out*/, float /*ratio*/, PVRow /*line_id*/)
{
	// The base filter does nothing
}

IMPL_FILTER(Picviz::PVLayerFilterHeatlineBase)


// "Colorize mode" filter

Picviz::PVLayerFilterHeatlineColor::PVLayerFilterHeatlineColor(PVCore::PVArgumentList const& l)
	: PVLayerFilterHeatlineBase(l)
{
	INIT_FILTER(PVLayerFilterHeatlineColor, l);
}

DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterHeatlineColor)
{
	return Picviz::PVLayerFilterHeatlineBase::default_args();
}

void Picviz::PVLayerFilterHeatlineColor::post(PVLayer& /*in*/, PVLayer& out, float ratio, PVRow line_id)
{
	const PVCore::PVHSVColor color((uint8_t)((float)(HSV_COLOR_RED-HSV_COLOR_GREEN)*ratio + (float)HSV_COLOR_GREEN));
	out.get_lines_properties().line_set_color(line_id, color);
}

IMPL_FILTER(Picviz::PVLayerFilterHeatlineColor)


// "Selection mode" filter

Picviz::PVLayerFilterHeatlineSel::PVLayerFilterHeatlineSel(PVCore::PVArgumentList const& l)
	: PVLayerFilterHeatlineBase(l)
{
	INIT_FILTER(PVLayerFilterHeatlineSel, l);
}

DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterHeatlineSel)
{
	PVCore::PVArgumentList args = Picviz::PVLayerFilterHeatlineBase::default_args();
	args[PVCore::PVArgumentKey(ARG_NAME_COLORS, ARG_DESC_COLORS)].setValue(PVCore::PVColorGradientDualSliderType());
	return args;
}

void Picviz::PVLayerFilterHeatlineSel::post(PVLayer& /*in*/, PVLayer& out, float ratio, PVRow line_id)
{
	PVCore::PVColorGradientDualSliderType ratios = _args[ARG_NAME_COLORS].value<PVCore::PVColorGradientDualSliderType>();

	const float *v = ratios.get_positions();

	float fmin = v[0];
	float fmax = v[1];

	if ((ratio > fmax) || (ratio < fmin)) {
		out.get_selection().set_line(line_id, 0);
	}
}

IMPL_FILTER(Picviz::PVLayerFilterHeatlineSel)

// "Select and colorize" mode
Picviz::PVLayerFilterHeatlineSelAndCol::PVLayerFilterHeatlineSelAndCol(PVCore::PVArgumentList const& l)
	: PVLayerFilterHeatlineBase(l)
{
	INIT_FILTER(PVLayerFilterHeatlineSelAndCol, l);
}

DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterHeatlineSelAndCol)
{
	PVCore::PVArgumentList args = Picviz::PVLayerFilterHeatlineBase::default_args();
	args[PVCore::PVArgumentKey(ARG_NAME_COLORS, ARG_DESC_COLORS)].setValue(PVCore::PVColorGradientDualSliderType());
	return args;
}

void Picviz::PVLayerFilterHeatlineSelAndCol::post(PVLayer& /*in*/, PVLayer& out, float ratio, PVRow line_id)
{
	// Colorize
	const PVCore::PVHSVColor color((uint8_t)((float)(HSV_COLOR_RED-HSV_COLOR_GREEN)*ratio + (float)HSV_COLOR_GREEN));
	out.get_lines_properties().line_set_color(line_id, color);

	// Select
	PVCore::PVColorGradientDualSliderType ratios = _args.value(ARG_NAME_COLORS).value<PVCore::PVColorGradientDualSliderType>();

	const float *v = ratios.get_positions();

	float fmin = v[0];
	float fmax = v[1];

	if ((ratio > fmax) || (ratio < fmin)) {
		out.get_selection().set_line(line_id, 0);
	}
}

IMPL_FILTER(Picviz::PVLayerFilterHeatlineSelAndCol)
