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

LibExport void picviz_filtering_function_init(void)
{
}

LibExport picviz_arguments_t *picviz_filtering_function_get_arguments(void)
{
	picviz_arguments_t *arguments;
	picviz_argument_item_t items[] = {
	  PICVIZ_ARGUMENTS_TEXTBOX(Axis, NULL, "")
	  PICVIZ_ARGUMENTS_TEXTBOX(Value, NULL, "")
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
	picviz_argument_item_t item_value;

	int line_id;

	int row = 0;
	pv_row nb_rows;
	int column = 0;
	pv_column nb_columns;

	char *axis_string;
	char *value_string;

	int axis_id;
	float value;

	plotted = view->PICVIZ_OBJECT_PLOTTED(parent);

	item_axis = picviz_arguments_get_item_from_name(arguments, "Axis");
	axis_string = picviz_arguments_item_get_string(item_axis);
	axis_id = atoi(axis_string);

	item_value = picviz_arguments_get_item_from_name(arguments, "Value");
	value_string = picviz_arguments_item_get_string(item_value);
	value = atof(value_string);

	nb_rows = picviz_view_get_row_count(view);
	for (row = 0; row < nb_rows; row++) {
	  if (picviz_plotted_get_value(plotted, row, axis_id) > value) {
	    picviz_selection_set_line(output_layer->selection, row, 1);
	  } else {
	    picviz_selection_set_line(output_layer->selection, row, 0);

	  }
	}


	return NULL;
}

LibExport void picviz_filtering_function_terminate(void)
{

}
