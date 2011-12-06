//! \file PVSelectionFilterScatterPlotSelectionSquare.cpp
//! $Id: PVSelectionFilterScatterPlotSelectionSquare.cpp 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <picviz/PVSelectionFilterScatterPlotSelectionSquare.h>
#include <pvkernel/core/PVColor.h>

/******************************************************************************
 *
 * Picviz::PVSelectionFilterScatterPlotSelectionSquare::PVSelectionFilterScatterPlotSelectionSquare
 *
 *****************************************************************************/
Picviz::PVSelectionFilterScatterPlotSelectionSquare::PVSelectionFilterScatterPlotSelectionSquare(PVCore::PVArgumentList const& l)
	: PVSelectionFilter(l)
{
	INIT_FILTER(PVSelectionFilterScatterPlotSelectionSquare, l);
}

/******************************************************************************
 *
 * DEFAULT_ARGS_FILTER(Picviz::PVSelectionFilterScatterPlotSelectionSquare)
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(Picviz::PVSelectionFilterScatterPlotSelectionSquare)
{
	PVCore::PVArgumentList args;
	args["x1_axis_index"] = 0;
	args["x2_axis_index"] = 1;
	args["x1_min"] = 0.0f;
	args["x1_max"] = 0.6f;
	args["x2_min"] = 0.0f;
	args["x2_max"] = 0.1f;
	args["Menu_Name"] = QString("Test");
	return args;
}

/******************************************************************************
 *
 * Picviz::PVSelectionFilterScatterPlotSelectionSquare::operator()
 *
 *****************************************************************************/
void Picviz::PVSelectionFilterScatterPlotSelectionSquare::operator()(PVSelection& in, PVSelection &out)
{
	float x1_min, x1_max, x2_min, x2_max;
	int x1_axis_index, x2_axis_index;
	int counter;
	int nb_lines;

	PVCore::PVColor color;
	QColor qcolor;

	//const PVSource* source = _view.get_source_parent();
	const PVPlotted* plotted = _view->get_plotted_parent();

	x1_axis_index = _args["x1_axis_index"].toInt();
	x2_axis_index = _args["x2_axis_index"].toInt();
	x1_min = _args["x1_min"].toFloat();
	x1_max = _args["x1_max"].toFloat();
	x2_min = _args["x2_min"].toFloat();
	x2_max = _args["x2_max"].toFloat();

	nb_lines = _view->get_qtnraw_parent().get_nrows();
// 
	for (counter = 0; counter < nb_lines; counter++) {
// 		if (_view->get_line_state_in_pre_filter_layer(counter)) {
		if (in.get_line(counter)) {
			float x1, x2;
// 
// 			//plotted_value = picviz_plotting_get_position(view->parent->parent, counter, axis_id);
			x1 = plotted->get_value(counter, x1_axis_index);
			x2 = plotted->get_value(counter, x2_axis_index);

			if ( (x1 > x1_min) && (x1 < x1_max) && (x2 > x2_min) && (x2 < x2_max)) {
				out.set_line(counter, true);
			} else {
				out.set_line(counter, false);
			}
// 
// 			qcolor.setHsvF((1.0f - plotted_value) / 3.0f, 1.0f, 1.0f);
// 			color.fromQColor(qcolor);
// 			out.get_lines_properties().line_set_rgb_from_color(counter, color);
		} else {
			out.set_line(counter, false);
		}
	}
}

IMPL_FILTER(Picviz::PVSelectionFilterScatterPlotSelectionSquare)
