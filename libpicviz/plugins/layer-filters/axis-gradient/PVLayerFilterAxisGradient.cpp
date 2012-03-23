//! \file PVLayerFilterAxisGradient.cpp
//! $Id: PVLayerFilterAxisGradient.cpp 2526 2011-05-02 12:21:26Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include "PVLayerFilterAxisGradient.h"
#include <pvkernel/core/PVColor.h>
#include <pvkernel/core/PVAxisIndexType.h>
#include <picviz/PVView.h>

#define ARG_NAME_AXIS "axis"
#define ARG_DESC_AXIS "Axis"

/******************************************************************************
 *
 * Picviz::PVLayerFilterAxisGradient::PVLayerFilterAxisGradient
 *
 *****************************************************************************/
Picviz::PVLayerFilterAxisGradient::PVLayerFilterAxisGradient(PVCore::PVArgumentList const& l)
	: PVLayerFilter(l)
{
	INIT_FILTER(PVLayerFilterAxisGradient, l);
	add_ctxt_menu_entry("Gradient on this axis", &PVLayerFilterAxisGradient::gradient_menu);
}

/******************************************************************************
 *
 * DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterAxisGradient)
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterAxisGradient)
{
	PVCore::PVArgumentList args;
	args[PVCore::PVArgumentKey(ARG_NAME_AXIS, QObject::tr(ARG_DESC_AXIS))].setValue(PVCore::PVAxisIndexType(0));
	return args;
}

/******************************************************************************
 *
 * Picviz::PVLayerFilterAxisGradient::operator()
 *
 *****************************************************************************/
void Picviz::PVLayerFilterAxisGradient::operator()(PVLayer& in, PVLayer &out)
{	
	int axis_id;

	PVCore::PVColor color;
	QColor qcolor;	

	//const PVSource* source = _view.get_source_parent();
	const PVPlotted* plotted = _view->get_plotted_parent();
	axis_id = _args[ARG_NAME_AXIS].value<PVCore::PVAxisIndexType>().get_original_index();

	PVPlotted::plotted_sub_col_t values_sel;
	float max_plotted,min_plotted;
	plotted->get_sub_col_minmax(values_sel, min_plotted, max_plotted, in.get_selection(), axis_id);
	PVPlotted::plotted_sub_col_t::const_iterator it;
	for (it = values_sel.begin(); it != values_sel.end(); it++) {
		if (should_cancel()) {
			if (&in != &out) {
				out = in;
			}
			return;
		}
		float plotted_value = it->second;
		PVRow line = it->first;

		qcolor.setHsvF((max_plotted-plotted_value)/(max_plotted-min_plotted) / 3.0f, 1.0f, 1.0f);
		color.fromQColor(qcolor);
		out.get_lines_properties().line_set_rgb_from_color(line, color);
	}
}

QList<PVCore::PVArgumentKey> Picviz::PVLayerFilterAxisGradient::get_args_keys_for_preset() const
{
	QList<PVCore::PVArgumentKey> keys = get_args_for_preset().keys();
	keys.removeAll(ARG_NAME_AXIS);
	return keys;
}

QString Picviz::PVLayerFilterAxisGradient::status_bar_description()
{
	return QString("Apply a gradient of color on a given axis.");
}

QString Picviz::PVLayerFilterAxisGradient::detailed_description()
{
	return QString("<b>Purpose</b><br/>This filter applies a color gradient on a wanted axis<hr><b>Behavior</b><br/>It will colorize with a gradient from green to red from the lowest axis value to the highest.");
}

PVCore::PVArgumentList Picviz::PVLayerFilterAxisGradient::gradient_menu(PVRow /*row*/, PVCol col, QString const& /*v*/)
{
	PVCore::PVArgumentList args;
	args[ARG_NAME_AXIS].setValue(PVCore::PVAxisIndexType(col));
	return args;

}

IMPL_FILTER(Picviz::PVLayerFilterAxisGradient)
