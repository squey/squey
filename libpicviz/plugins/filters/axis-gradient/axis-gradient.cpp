/*
 * $Id$
 * Copyright (C) Sebastien Tricaud 2010
 * Copyright (C) Philippe Saade 2010
 * 
 */

#include <stdlib.h>

#include <pcre.h>

#include <Qt/qcolor.h>

#include <picviz/general.h>
#include <picviz/debug.h>
#include <picviz/view.h>
#include <picviz/layer.h>
#include <picviz/source.h>
#include <picviz/plotted.h>
#include <picviz/selection.h>
#include <picviz/arguments.h>
#include <picviz/filters.h>
#include <picviz/utils.h>

LibCPPExport void picviz_filtering_function_init(void) {}

LibCPPExport picviz_arguments_t *picviz_filtering_function_get_arguments(void)
{
	picviz_arguments_t *arguments;
	picviz_argument_item_t items[] = {
	  PICVIZ_ARGUMENTS_AXIS(Axis, 0),
	  PICVIZ_ARGUMENTS_END
	};

	arguments = picviz_arguments_new();
	picviz_arguments_item_list_append(arguments, items);

	return arguments;
}

LibCPPExport picviz_filter_type_t picviz_filtering_function_get_type(void)
{
	return PICVIZ_FILTER_NOCONFIG;
}

LibCPPExport char *picviz_filtering_function_exec(picviz_view_t *view, picviz_layer_t * /*input_layer*/, picviz_layer_t *output_layer, picviz_arguments_t *arguments)
{
	picviz_source_t *source;
	picviz_plotted_t *plotted;
	//picviz_argument_item_t item;

	//int line_id;

	//int row = 0;
	//pv_row nb_rows;
	//int column = 0;
	//pv_column nb_columns;
	//char *textbox_string;

	picviz_argument_item_t item_axis;
	//char *axis_string;
	int axis_id;

	int nb_lines;
	int counter;

	Picviz::Color color;
	QColor qcolor;

	source = picviz_view_get_source_parent(view);
	plotted = view->PICVIZ_OBJECT_PLOTTED(parent);

	/* picviz_arguments_debug(arguments); */

	item_axis = picviz_arguments_get_item_from_name(arguments, "Axis");
	axis_id = item_axis.ival;

	nb_lines = picviz_array_get_elts_count(view->PICVIZ_OBJECT_PLOTTED(parent)->PICVIZ_OBJECT_PLOTTING(table));

	for (counter = 0; counter < nb_lines; counter++) {
		float plotted_value;

		//plotted_value = picviz_plotting_get_position(view->parent->parent, counter, axis_id);
		plotted_value = picviz_plotted_get_value(view->parent, counter, axis_id);

		qcolor.setHsvF((1.0f - plotted_value) / 3.0f, 1.0f, 1.0f);
		color.fromQColor(qcolor);
		picviz_lines_properties_line_set_rgb_from_color(output_layer->lines_properties, counter, color);
	}
	return NULL;
}

LibCPPExport void picviz_filtering_function_terminate(void) {}
