/*
 * $Id$
 * Copyright (C) Sebastien Tricaud 2010
 * Copyright (C) Philippe Saade 2010
 * 
 */

#include <stdlib.h>

#include <picviz/general.h>
#include <picviz/debug.h>
#include <picviz/view.h>
#include <picviz/layer.h>
#include <picviz/source.h>
#include <picviz/plotted.h>
#include <picviz/selection.h>
#include <picviz/format.h>
#include <picviz/arguments.h>
#include <picviz/filters.h>

LibExport void picviz_filtering_function_init(void) {}

LibExport picviz_arguments_t *picviz_filtering_function_get_arguments(void)
{
	picviz_arguments_t *arguments;
	picviz_argument_item_t items[] = {
	  PICVIZ_ARGUMENTS_AXIS(Axis, 0)
	  PICVIZ_ARGUMENTS_END
	};

	arguments = picviz_arguments_new();
	picviz_arguments_item_list_append(arguments, items);

	return arguments;
}

LibExport picviz_filter_type_t picviz_filtering_function_get_type(void)
{
	return PICVIZ_FILTER_NOCONFIG;
}

LibExport char *picviz_filtering_function_exec(picviz_view_t *view, picviz_layer_t *input_layer, picviz_layer_t *output_layer, picviz_arguments_t *arguments)
{
	picviz_plotted_t *plotted;
	picviz_argument_item_t item_axis;

	int line_id;

	int row = 0;
	pv_row nb_rows;
	int column = 0;
	pv_column nb_columns;

	int axis_id;
	int i;
	int trojan_ports[17] = {1080, 2283, 2535, 2745, 3127, 3128, 3410, 5554, 8866, 9898, 10000, 10080, 12345, 17300, 27374, 65506, 0};


	plotted = view->PICVIZ_OBJECT_PLOTTED(parent);

	item_axis = picviz_arguments_get_item_from_name(arguments, "Axis");
	axis_id = item_axis.ival;


	picviz_selection_A2A_select_none(output_layer->selection);
	nb_rows = picviz_view_get_row_count(view);
	for (row = 0; row < nb_rows; row++) {
		char *nraw_string = picviz_view_get_data(view, row, axis_id);
		int port_int = atoi(nraw_string);

		for (i=0; trojan_ports[i] != 0; i++) {
		  if (port_int == trojan_ports[i]) {
		        picviz_selection_set_line(output_layer->selection, row, 1);
		  }
		}
	}

	return NULL;
}

LibExport void picviz_filtering_function_terminate(void) {}
