/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvbase/qhashes.h>

#include "PVLayerFilterHeatline.h"

#include <pvkernel/core/inendi_bench.h>
#include <pvkernel/core/PVColor.h>
#include <pvkernel/core/PVAxisIndexType.h>
#include <pvkernel/core/PVPercentRangeType.h>
#include <pvkernel/core/PVEnumType.h>
#include <pvkernel/rush/PVUtils.h>

#include <inendi/PVView.h>

#include <tbb/concurrent_hash_map.h>
#include <tbb/enumerable_thread_specific.h>

#include <math.h>
#include <unordered_map>

#include <omp.h>

#define ARG_NAME_AXES "axes"
#define ARG_DESC_AXES "Axis"
#define ARG_NAME_SCALE "scale"
#define ARG_DESC_SCALE "Scale factor"
#define ARG_NAME_COLORS "colors"
#define ARG_DESC_COLORS "Frequency range"

/******************************************************************************
 *
 * Inendi::PVLayerFilterHeatlineBase::PVLayerFilterHeatlineBase
 *
 *****************************************************************************/
Inendi::PVLayerFilterHeatlineBase::PVLayerFilterHeatlineBase(PVCore::PVArgumentList const& l)
	: PVLayerFilter(l)
{
	INIT_FILTER(PVLayerFilterHeatlineBase, l);
}

/******************************************************************************
 *
 * DEFAULT_ARGS_FILTER(Inendi::PVLayerFilterHeatlineBase)
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(Inendi::PVLayerFilterHeatlineBase)
{
	PVCore::PVArgumentList args;
	args[PVCore::PVArgumentKey(ARG_NAME_AXES, ARG_DESC_AXES)].setValue(PVCore::PVAxisIndexType());

	PVCore::PVEnumType scale(QStringList() << "Linear" << "Log", 0);
	args[PVCore::PVArgumentKey(ARG_NAME_SCALE, ARG_DESC_SCALE)].setValue(scale);

	return args;
}

/******************************************************************************
 *
 * Inendi::PVLayerFilterHeatlineBase::get_default_args_for_view
 *
 *****************************************************************************/
PVCore::PVArgumentList Inendi::PVLayerFilterHeatlineBase::get_default_args_for_view(PVView const&)
{
	PVCore::PVArgumentList args = get_default_args();
	// Default args with the "key" tag
	args[ARG_NAME_AXES].setValue(PVCore::PVAxisIndexType(0));
	return args;
}

/******************************************************************************
 *
 * Inendi::PVLayerFilterHeatlineBase::operator()
 *
 *****************************************************************************/
void Inendi::PVLayerFilterHeatlineBase::operator()(PVLayer& in, PVLayer &out)
{
	BENCH_START(heatline);

	PVRush::PVNraw const& nraw = _view->get_rushnraw_parent();

	PVCore::PVAxisIndexType axis = _args[ARG_NAME_AXES].value<PVCore::PVAxisIndexType>();
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

	PVCore::PVPercentRangeType ratios = _args[ARG_NAME_COLORS].value<PVCore::PVPercentRangeType>();

	const double *freq_values = ratios.get_values();

	const double freq_min = freq_values[0];
	const double freq_max = freq_values[1];

	bool bLog = _args[ARG_NAME_SCALE].value<PVCore::PVEnumType>().get_sel().compare("Log") == 0;

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
				it = freqs.emplace(std::move(tmp_str), 1).first;
			}
			else {
				it->second++;
			}
			row_values[r] = &it->second;
		},
		in.get_selection());

	lines_hash_t::const_iterator it;
	PVRow max_n = 0;
	PVRow min_n = 0xFFFFFFFF;
	for (it = freqs.begin(); it != freqs.end(); it++) {
		//std::cout << it->first << ": " << it->second << std::endl;
		const PVRow cur_n = it->second;
		if (cur_n > max_n) {
			max_n = cur_n;
		}
		if (cur_n < min_n) {
			min_n = cur_n;
		}
	}
	assert(min_n <= max_n);

	if (max_n == min_n) {
		in.get_selection().visit_selected_lines(
			[&](const PVRow r)
			{
				this->post(in, out, 1.0 / (double)freqs.size(),
				           freq_min, freq_max, r);
			}, nrows);
	}
	else {
		const double diff = max_n - min_n;
		const double log_diff = log(diff);

		in.get_selection().visit_selected_lines(
			[&](const PVRow r)
			{
				assert(r < row_values.size());
				const PVRow *pfreq = row_values[r];
				// AG: fixme: that should be an assert
				if (!pfreq) {
					return;
				}
				const PVRow freq = *pfreq;
				double ratio;
				if (bLog) {
					if (freq == min_n) {
						ratio = 0;
					}
					else {
						ratio = log(freq-min_n)/log_diff;
					}
				}
				else {
					ratio = (double)(freq-min_n)/diff;
				}
				//std::cout << "line " << r << ", n=" << freq << ", ratio=" << std::setprecision(7) << ratio << std::endl;
				this->post(in, out, ratio, freq_min, freq_max, r);
			}, nrows);
	}

	BENCH_END(heatline, "heatline", 1, 1, sizeof(PVRow), nrows);
}

PVCore::PVArgumentKeyList Inendi::PVLayerFilterHeatlineBase::get_args_keys_for_preset() const
{
	PVCore::PVArgumentKeyList keys = get_default_args().keys();
	keys.erase(std::find(keys.begin(), keys.end(), ARG_NAME_AXES));
	return keys;
}

void Inendi::PVLayerFilterHeatlineBase::post(const PVLayer& /*in*/, PVLayer& /*out*/,
                                             const double /*ratio*/,
                                             const double /*fmin*/, const double /*fmax*/,
                                             const PVRow /*line_id*/)
{
	// The base filter does nothing
}

IMPL_FILTER(Inendi::PVLayerFilterHeatlineBase)


// "Colorize mode" filter

Inendi::PVLayerFilterHeatlineColor::PVLayerFilterHeatlineColor(PVCore::PVArgumentList const& l)
	: PVLayerFilterHeatlineBase(l)
{
	INIT_FILTER(PVLayerFilterHeatlineColor, l);
}

DEFAULT_ARGS_FILTER(Inendi::PVLayerFilterHeatlineColor)
{
	return Inendi::PVLayerFilterHeatlineBase::default_args();
}

void Inendi::PVLayerFilterHeatlineColor::post(const PVLayer& /*in*/, PVLayer& out,
                                              const double ratio,
                                              const double /*fmin*/, const double /*fmax*/,
                                              const PVRow line_id)
{
	const PVCore::PVHSVColor color((uint8_t)((double)(HSV_COLOR_RED-HSV_COLOR_GREEN)*ratio + (double)HSV_COLOR_GREEN));
	out.get_lines_properties().line_set_color(line_id, color);
}

IMPL_FILTER(Inendi::PVLayerFilterHeatlineColor)


// "Selection mode" filter

Inendi::PVLayerFilterHeatlineSel::PVLayerFilterHeatlineSel(PVCore::PVArgumentList const& l)
	: PVLayerFilterHeatlineBase(l)
{
	INIT_FILTER(PVLayerFilterHeatlineSel, l);
}

DEFAULT_ARGS_FILTER(Inendi::PVLayerFilterHeatlineSel)
{
	PVCore::PVArgumentList args = Inendi::PVLayerFilterHeatlineBase::default_args();
	args[PVCore::PVArgumentKey(ARG_NAME_COLORS, ARG_DESC_COLORS)].setValue(PVCore::PVPercentRangeType());
	return args;
}

void Inendi::PVLayerFilterHeatlineSel::post(const PVLayer& /*in*/, PVLayer& out,
                                            const double ratio,
                                            const double fmin, const double fmax,
                                            const PVRow line_id)
{
	if ((ratio >= fmin) && (ratio <= fmax)) {
		out.get_selection().set_line(line_id, 0);
	}
}

IMPL_FILTER(Inendi::PVLayerFilterHeatlineSel)

// "Select and colorize" mode
Inendi::PVLayerFilterHeatlineSelAndCol::PVLayerFilterHeatlineSelAndCol(PVCore::PVArgumentList const& l)
	: PVLayerFilterHeatlineBase(l)
{
	INIT_FILTER(PVLayerFilterHeatlineSelAndCol, l);
}

DEFAULT_ARGS_FILTER(Inendi::PVLayerFilterHeatlineSelAndCol)
{
	PVCore::PVArgumentList args = Inendi::PVLayerFilterHeatlineBase::default_args();
	args[PVCore::PVArgumentKey(ARG_NAME_COLORS, ARG_DESC_COLORS)].setValue(PVCore::PVPercentRangeType());
	return args;
}

void Inendi::PVLayerFilterHeatlineSelAndCol::post(const PVLayer& /*in*/, PVLayer& out,
                                                  const double ratio,
                                                  const double fmin, const double fmax,
                                                  const PVRow line_id)
{
	// Colorize
	const PVCore::PVHSVColor color((uint8_t)((double)(HSV_COLOR_RED-HSV_COLOR_GREEN)*ratio + (double)HSV_COLOR_GREEN));
	out.get_lines_properties().line_set_color(line_id, color);

	// Select
	if ((ratio < fmin) || (ratio > fmax)) {
		out.get_selection().set_line(line_id, 0);
	}
}

IMPL_FILTER(Inendi::PVLayerFilterHeatlineSelAndCol)