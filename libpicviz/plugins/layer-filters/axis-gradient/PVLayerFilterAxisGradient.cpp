//! \file PVLayerFilterAxisGradient.cpp
//! $Id: PVLayerFilterAxisGradient.cpp 2526 2011-05-02 12:21:26Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include "PVLayerFilterAxisGradient.h"
#include <picviz/PVColor.h>
#include <pvcore/PVAxisIndexType.h>

/******************************************************************************
 *
 * Picviz::PVLayerFilterAxisGradient::PVLayerFilterAxisGradient
 *
 *****************************************************************************/
Picviz::PVLayerFilterAxisGradient::PVLayerFilterAxisGradient(PVCore::PVArgumentList const& l)
	: PVLayerFilter(l)
{
	INIT_FILTER(PVLayerFilterAxisGradient, l);
}

/******************************************************************************
 *
 * DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterAxisGradient)
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterAxisGradient)
{
	PVCore::PVArgumentList args;
	args["Axis"].setValue(PVCore::PVAxisIndexType(0));
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
	int counter;
	int nb_lines;

	PVColor color;
	QColor qcolor;	

	//const PVSource* source = _view.get_source_parent();
	const PVPlotted_p plotted = _view->get_plotted_parent();
	
	axis_id = _args["Axis"].value<PVCore::PVAxisIndexType>().get_original_index();
	axis_id = _view->axes_combination.get_axis_column_index(axis_id);
	nb_lines = _view->get_qtnraw_parent().size();
	
	//PVRow size_sel = in.get_selection().get_number_of_selected_lines_in_range(0, plotted->table.size() - 1);
	for (counter = 0; counter < nb_lines; counter++) {
		if (_view->get_line_state_in_pre_filter_layer(counter)) {
			float plotted_value;

			//plotted_value = picviz_plotting_get_position(view->parent->parent, counter, axis_id);
			plotted_value = plotted->get_value(counter, axis_id);

			qcolor.setHsvF((1.0f - plotted_value) / 3.0f, 1.0f, 1.0f);
			color.fromQColor(qcolor);
			out.get_lines_properties().line_set_rgb_from_color(counter, color);
		}
	}
}

QString Picviz::PVLayerFilterAxisGradient::status_bar_description()
{
	return QString("Apply a gradient of color on a given axis.");
}

QString Picviz::PVLayerFilterAxisGradient::detailed_description()
{
	return QString("<b>Purpose</b><br/>This filter applies a color gradient on a wanted axis<hr><b>Behavior</b><br/>It will colorize with a gradient from green to red from the lowest axis value to the highest.");
}

IMPL_FILTER(Picviz::PVLayerFilterAxisGradient)
