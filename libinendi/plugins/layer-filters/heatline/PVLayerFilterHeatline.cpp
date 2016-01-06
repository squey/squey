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

#include <cmath>
#include <unordered_map>

#define ARG_NAME_AXES "axes"
#define ARG_DESC_AXES "Axis"
#define ARG_NAME_SCALE "scale"
#define ARG_DESC_SCALE "Scale factor"
#define ARG_NAME_COLORS "colors"
#define ARG_DESC_COLORS "Frequency range"

/******************************************************************************
 *
 * Inendi::PVLayerFilterHeatline::PVLayerFilterHeatline
 *
 *****************************************************************************/
Inendi::PVLayerFilterHeatline::PVLayerFilterHeatline(PVCore::PVArgumentList const& l)
	: PVLayerFilter(l)
{
	INIT_FILTER(PVLayerFilterHeatline, l);
}

/******************************************************************************
 *
 * Inendi::PVLayerFilterHeatline::get_default_args_for_view
 *
 *****************************************************************************/
PVCore::PVArgumentList Inendi::PVLayerFilterHeatline::get_default_args_for_view(PVView const&)
{
	PVCore::PVArgumentList args = get_default_args();
	// Default args with the "key" tag
	args[ARG_NAME_AXES].setValue(PVCore::PVAxisIndexType(0));
	return args;
}

/******************************************************************************
 *
 * Inendi::PVLayerFilterHeatline::operator()
 *
 *****************************************************************************/
void Inendi::PVLayerFilterHeatline::operator()(PVLayer& in, PVLayer &out)
{
	BENCH_START(heatline);

	// Extract Nraw data
	PVRush::PVNraw const& nraw = _view->get_rushnraw_parent();

	// Extract axis where we apply heatline computation
	PVCore::PVAxisIndexType axis = _args[ARG_NAME_AXES].value<PVCore::PVAxisIndexType>();
	const PVCol axis_id = axis.get_original_index();

	// Extract ratio information
	PVCore::PVPercentRangeType ratios = _args[ARG_NAME_COLORS].value<PVCore::PVPercentRangeType>();

	const double *freq_values = ratios.get_values();

	const double freq_min = freq_values[0];
	const double freq_max = freq_values[1];

	// Extract scale information.
	bool bLog = _args[ARG_NAME_SCALE].value<PVCore::PVEnumType>().get_sel().compare("Log") == 0;

	// Default to the original selection
	out.get_selection() = in.get_selection();

	// Per-thread frequencies
	typedef std::unordered_map<std::string_tbb, PVRow> lines_hash_t;
	lines_hash_t freqs;

	// Get full number of row
	const PVRow nrows = _view->get_row_count();
	freqs.reserve(nrows);

	std::vector<PVRow const*> row_values;
	row_values.resize(nrows, nullptr);

	// Count number of occurance for each value in choosen axis.
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

	// Compute min/max values for every frequency.
	PVRow max_n = std::numeric_limits<PVRow>::lowest();
	PVRow min_n = std::numeric_limits<PVRow>::max();
	for (auto & freq: freqs) {
		const PVRow cur_n = freq.second;
		if (cur_n > max_n) {
			max_n = cur_n;
		}
		if (cur_n < min_n) {
			min_n = cur_n;
		}
	}
	assert(min_n <= max_n && "We should have a correct order between min/max");

	if (max_n == min_n) {
		in.get_selection().visit_selected_lines(
			[&](const PVRow r)
			{
				this->post(out, 1.0 / (double)freqs.size(),
				           freq_min, freq_max, r);
			}, nrows);
	}
	else {
		const double diff = max_n - min_n;
		const double log_diff = std::log(diff);

		in.get_selection().visit_selected_lines(
			[&](const PVRow r)
			{
				assert(r < row_values.size());
				const PVRow *pfreq = row_values[r];

				assert(pfreq && "This should be filled by frequency computation");

				const PVRow freq = *pfreq;

				// Computation ratio to havec 1 for freq = max_n and 0 for freq = min_n
				double ratio;
				if (bLog) {
					if (freq == min_n) {
						ratio = 0;
					}
					else {
						ratio = std::log(freq-min_n)/log_diff;
					}
				}
				else {
					ratio = (double)(freq-min_n)/diff;
				}

				this->post(out, ratio, freq_min, freq_max, r);
			}, nrows);
	}

	BENCH_END(heatline, "heatline", 1, 1, sizeof(PVRow), nrows);
}

PVCore::PVArgumentKeyList Inendi::PVLayerFilterHeatline::get_args_keys_for_preset() const
{
	// Sve everything but axis in the preset.
	PVCore::PVArgumentKeyList keys = get_default_args().keys();
	keys.erase(std::find(keys.begin(), keys.end(), ARG_NAME_AXES));
	return keys;
}

DEFAULT_ARGS_FILTER(Inendi::PVLayerFilterHeatline)
{
	PVCore::PVArgumentList args;

	PVCore::PVEnumType scale(QStringList() << "Linear" << "Log", 0);

	args[PVCore::PVArgumentKey(ARG_NAME_SCALE, ARG_DESC_SCALE)].setValue(scale);
	args[PVCore::PVArgumentKey(ARG_NAME_AXES, ARG_DESC_AXES)].setValue(PVCore::PVAxisIndexType());
	args[PVCore::PVArgumentKey(ARG_NAME_COLORS, ARG_DESC_COLORS)].setValue(PVCore::PVPercentRangeType());
	return args;
}

void Inendi::PVLayerFilterHeatline::post(PVLayer& out,
                                         const double ratio,
                                         const double fmin, const double fmax,
                                         const PVRow line_id)
{
	// Colorize line dpeending on ratio value. (High ration -> red, low ration -> green)
	const PVCore::PVHSVColor color((uint8_t)((double)(HSV_COLOR_RED-HSV_COLOR_GREEN)*ratio + (double)HSV_COLOR_GREEN));
	out.get_lines_properties().line_set_color(line_id, color);

	// UnSelect line out of min/max choosen frequency.
	if ((ratio < fmin) || (ratio > fmax)) {
		out.get_selection().set_line(line_id, 0);
	}
}

IMPL_FILTER(Inendi::PVLayerFilterHeatline)
