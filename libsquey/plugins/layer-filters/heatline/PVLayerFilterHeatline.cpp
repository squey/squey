//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include "PVLayerFilterHeatline.h"

#include <squey/PVView.h>

#include <pvkernel/core/squey_bench.h>
#include <pvkernel/core/PVOriginalAxisIndexType.h>
#include <pvkernel/core/PVPercentRangeType.h>
#include <pvkernel/core/PVEnumType.h>
#include <pvkernel/rush/PVUtils.h>
#include <pvkernel/core/PVAlgorithms.h>

#include <pvcop/db/algo.h>

#include <cmath>
#include <unordered_map>

#define ARG_NAME_AXIS "axis"
#define ARG_DESC_AXIS "Axis"
#define ARG_NAME_SCALE "scale"
#define ARG_DESC_SCALE "Scale factor"
#define ARG_NAME_COLORS "colors"
#define ARG_DESC_COLORS "Frequency range"

/******************************************************************************
 *
 * Squey::PVLayerFilterHeatline::PVLayerFilterHeatline
 *
 *****************************************************************************/
Squey::PVLayerFilterHeatline::PVLayerFilterHeatline(PVCore::PVArgumentList const& l)
    : PVLayerFilter(l)
{
	INIT_FILTER(PVLayerFilterHeatline, l);
}

/******************************************************************************
 *
 * Squey::PVLayerFilterHeatline::get_default_args_for_view
 *
 *****************************************************************************/
PVCore::PVArgumentList Squey::PVLayerFilterHeatline::get_default_args_for_view(PVView const&)
{
	PVCore::PVArgumentList args = get_default_args();
	args[ARG_NAME_AXIS].setValue(PVCore::PVOriginalAxisIndexType(PVCol(0)));
	return args;
}

/******************************************************************************
 *
 * Squey::PVLayerFilterHeatline::operator()
 *
 *****************************************************************************/
void Squey::PVLayerFilterHeatline::operator()(PVLayer const& in, PVLayer& out)
{
	// Nothing to do if selection is empty
	if (in.get_selection().bit_count() == 0) {
		return;
	}

	// Extract Nraw data
	PVRush::PVNraw const& nraw = _view->get_rushnraw_parent();

	// Extract axis where we apply heatline computation
	auto axis = _args[ARG_NAME_AXIS].value<PVCore::PVOriginalAxisIndexType>();
	const PVCol axis_id = axis.get_original_index();

	pvlogger::info() << "axis_id=" << axis_id << std::endl;

	// Extract ratio information
	auto ratios = _args[ARG_NAME_COLORS].value<PVCore::PVPercentRangeType>();

	const double* freq_values = ratios.get_values();

	const double freq_min = freq_values[0];
	const double freq_max = freq_values[1];

	// Extract scale information.
	bool bLog = _args[ARG_NAME_SCALE].value<PVCore::PVEnumType>().get_sel().compare("Log") == 0;

	// Default to the original selection
	out.get_selection() = in.get_selection();

	// Count number of occurance for each value in choosen axis.
	pvcop::db::array const& col = nraw.column(axis_id);

	pvcop::db::groups group;
	pvcop::db::extents extents;
	const Squey::PVSelection& sel = out.get_selection();

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

PVCore::PVArgumentKeyList Squey::PVLayerFilterHeatline::get_args_keys_for_preset() const
{
	// Sve everything but axis in the preset.
	PVCore::PVArgumentKeyList keys = get_default_args().keys();
	keys.erase(std::find(keys.begin(), keys.end(), ARG_NAME_AXIS));
	return keys;
}

DEFAULT_ARGS_FILTER(Squey::PVLayerFilterHeatline)
{
	PVCore::PVArgumentList args;

	PVCore::PVEnumType scale(QStringList() << "Linear"
	                                       << "Log",
	                         0);

	args[PVCore::PVArgumentKey(ARG_NAME_SCALE, ARG_DESC_SCALE)].setValue(scale);
	args[PVCore::PVArgumentKey(ARG_NAME_AXIS, ARG_DESC_AXIS)].setValue(PVCore::PVOriginalAxisIndexType());
	args[PVCore::PVArgumentKey(ARG_NAME_COLORS, ARG_DESC_COLORS)].setValue(
	    PVCore::PVPercentRangeType());
	return args;
}

void Squey::PVLayerFilterHeatline::post(
    PVLayer& out, const double ratio, const double fmin, const double fmax, const PVRow line_id)
{
	// Colorize line dpeending on ratio value. (High ratio -> red, low ratio -> green)
	const PVCore::PVHSVColor color((uint8_t)(
	    (double)(HSV_COLOR_RED.h() - HSV_COLOR_BLUE.h()) * ratio + (double)HSV_COLOR_BLUE.h()));
	out.get_lines_properties().set_line_properties(line_id, color);

	// UnSelect line out of min/max choosen frequency.
	if ((ratio < fmin) || (ratio > fmax)) {
		out.get_selection().set_line(line_id, false);
	}
}
