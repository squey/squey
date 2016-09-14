/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVLayerFilterHeatline.h"

#include <inendi/PVView.h>

#include <pvkernel/core/inendi_bench.h>
#include <pvkernel/core/PVAxisIndexType.h>
#include <pvkernel/core/PVPercentRangeType.h>
#include <pvkernel/core/PVEnumType.h>
#include <pvkernel/rush/PVUtils.h>
#include <pvkernel/core/PVAlgorithms.h>

#include <pvcop/db/algo.h>

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
void Inendi::PVLayerFilterHeatline::operator()(PVLayer const& in, PVLayer& out)
{
	// Nothing to do if selection is empty
	if (in.get_selection().bit_count() == 0) {
		return;
	}

	// Extract Nraw data
	PVRush::PVNraw const& nraw = _view->get_rushnraw_parent();

	// Extract axis where we apply heatline computation
	PVCore::PVAxisIndexType axis = _args[ARG_NAME_AXES].value<PVCore::PVAxisIndexType>();
	const PVCol axis_id = axis.get_original_index();

	// Extract ratio information
	PVCore::PVPercentRangeType ratios = _args[ARG_NAME_COLORS].value<PVCore::PVPercentRangeType>();

	const double* freq_values = ratios.get_values();

	const double freq_min = freq_values[0];
	const double freq_max = freq_values[1];

	// Extract scale information.
	bool bLog = _args[ARG_NAME_SCALE].value<PVCore::PVEnumType>().get_sel().compare("Log") == 0;

	// Default to the original selection
	out.get_selection() = in.get_selection();

	// Count number of occurance for each value in choosen axis.
	pvcop::db::array const& col = nraw.collection().column(axis_id);

	pvcop::db::groups group;
	pvcop::db::extents extents;
	pvcop::db::selection& pvsel = out.get_selection();

	// Set correct size to selection.
	pvcop::db::selection sel(pvsel, 0, col.size());

	col.group(group, extents, sel);

	pvcop::db::array count = col.group_count(group, extents);

	// Compute min and max value
	pvcop::db::array mm = pvcop::db::algo::minmax(count);
	auto& minmax = mm.to_core_array<pvcop::db::indexes::type>();
	pvcop::db::indexes::type min_n = minmax[0];
	pvcop::db::indexes::type max_n = minmax[1];

	assert(min_n <= max_n && "We should have a correct order between min/max");

	if (max_n == min_n) {
		// Case where every value have the same frequency.
		// Set the same color depending on the number of value
		for (auto it = sel.begin(); it != sel.end(); ++it) {
			post(out, 1.0 / extents.size(), freq_min, freq_max, it.index());
		}
	} else {
		auto const& group_array = group.to_core_array();
		auto const& count_array = count.to_core_array<pvcop::db::indexes::type>();

		size_t index = 0;
		size_t selected_index = 0;
		// FIXME : We should implement a pvcop::db::array::sel_iterator to iterate
		//         more efficiently on selected rows
		for (auto it = sel.begin(); it != sel.end(); ++it, index++) {
			if (not*it) {
				continue;
			}
			double cum = count_array[group_array[selected_index++]];

			// Computation ratio to havec 1 for freq = max_n and 0 for freq = min_n
			double ratio;
			if (bLog) {
				ratio = PVCore::log_scale<double>(cum, min_n, max_n);
			} else {
				ratio = (cum - min_n) / (max_n - min_n);
			}
			post(out, ratio, freq_min, freq_max, index);
		}
	}
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

	PVCore::PVEnumType scale(QStringList() << "Linear"
	                                       << "Log",
	                         0);

	args[PVCore::PVArgumentKey(ARG_NAME_SCALE, ARG_DESC_SCALE)].setValue(scale);
	args[PVCore::PVArgumentKey(ARG_NAME_AXES, ARG_DESC_AXES)].setValue(PVCore::PVAxisIndexType());
	args[PVCore::PVArgumentKey(ARG_NAME_COLORS, ARG_DESC_COLORS)].setValue(
	    PVCore::PVPercentRangeType());
	return args;
}

void Inendi::PVLayerFilterHeatline::post(
    PVLayer& out, const double ratio, const double fmin, const double fmax, const PVRow line_id)
{
	// Colorize line dpeending on ratio value. (High ratio -> red, low ratio -> green)
	const PVCore::PVHSVColor color(
	    (uint8_t)((double)(HSV_COLOR_RED - HSV_COLOR_GREEN) * ratio + (double)HSV_COLOR_GREEN));
	out.get_lines_properties().set_line_properties(line_id, color);

	// UnSelect line out of min/max choosen frequency.
	if ((ratio < fmin) || (ratio > fmax)) {
		out.get_selection().set_line(line_id, 0);
	}
}
