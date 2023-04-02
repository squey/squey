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

#include "PVLayerFilterAxisGradient.h"
#include <pvkernel/core/PVOriginalAxisIndexType.h>
#include <inendi/PVPlotted.h>

#define ARG_NAME_AXIS "axis"
#define ARG_DESC_AXIS "Axis"

/******************************************************************************
 *
 * Inendi::PVLayerFilterAxisGradient::PVLayerFilterAxisGradient
 *
 *****************************************************************************/
Inendi::PVLayerFilterAxisGradient::PVLayerFilterAxisGradient(PVCore::PVArgumentList const& l)
    : PVLayerFilter(l)
{
	INIT_FILTER(PVLayerFilterAxisGradient, l);
}

/******************************************************************************
 *
 * DEFAULT_ARGS_FILTER(Inendi::PVLayerFilterAxisGradient)
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(Inendi::PVLayerFilterAxisGradient)
{
	PVCore::PVArgumentList args;
	args[PVCore::PVArgumentKey(ARG_NAME_AXIS, QObject::tr(ARG_DESC_AXIS))].setValue(
	    PVCore::PVOriginalAxisIndexType(PVCol(0)));
	return args;
}

/******************************************************************************
 *
 * Inendi::PVLayerFilterAxisGradient::operator()
 *
 *****************************************************************************/
void Inendi::PVLayerFilterAxisGradient::operator()(PVLayer const& in, PVLayer& out)
{
	PVCol axis_id;

	PVCore::PVHSVColor color;

	const PVPlotted& plotted = _view->get_parent<PVPlotted>();
	axis_id = _args[ARG_NAME_AXIS].value<PVCore::PVOriginalAxisIndexType>().get_original_index();

	PVRow r_max, r_min;
	plotted.get_col_minmax(r_min, r_max, in.get_selection(), axis_id);
	const uint32_t min_plotted = (plotted.get_value(r_min, axis_id));
	const uint32_t max_plotted = (plotted.get_value(r_max, axis_id));
	PVLOG_INFO("PVLayerFilterAxisGradient: min/max = %u/%u\n", min_plotted, max_plotted);
	const double diff = max_plotted - min_plotted;
	in.get_selection().visit_selected_lines(
	    [&](PVRow const r) {
		    const uint32_t plotted_value = (plotted.get_value(r, axis_id));

		    PVCore::PVHSVColor color;
		    // From green to red.. !
		    color = PVCore::PVHSVColor(
		        HSV_COLOR_RED.h() - ((uint8_t)(((double)(plotted_value - min_plotted) / diff) *
		                                       (double)(HSV_COLOR_RED.h() - HSV_COLOR_BLUE.h()))));
		    out.get_lines_properties().set_line_properties(r, color);
	    },
	    _view->get_row_count());
}

std::vector<PVCore::PVArgumentKey>
Inendi::PVLayerFilterAxisGradient::get_args_keys_for_preset() const
{
	PVCore::PVArgumentKeyList keys = get_default_args().keys();
	keys.erase(std::find(keys.begin(), keys.end(), ARG_NAME_AXIS));
	return keys;
}

QString Inendi::PVLayerFilterAxisGradient::status_bar_description()
{
	return {"Apply a gradient of color on a given axis."};
}

QString Inendi::PVLayerFilterAxisGradient::detailed_description()
{
	return {"<b>Purpose</b><br/>This filter applies a color gradient on a wanted "
	               "axis<hr><b>Behavior</b><br/>It will colorize with a gradient from green to red "
	               "from the lowest axis value to the highest."};
}
