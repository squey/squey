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
	return NULL;
}

LibExport picviz_filter_type_t picviz_filtering_function_get_type(void)
{
	return PICVIZ_FILTER_NOCONFIG;
}

LibExport char *picviz_filtering_function_exec(picviz_view_t *view, picviz_layer_t *input_layer, picviz_layer_t *output_layer, picviz_arguments_t *arguments)
{
	picviz_source_t *source;
	picviz_plotted_t *plotted;

	int line_id, j, k;
	int quarter;

	float *diff_keys;
	float key;

	int found;

	source = picviz_view_get_source_parent(view);
	plotted = view->PICVIZ_OBJECT_PLOTTED(parent);
	quarter = plotted->table->nelts / 4;

/* 	picviz_selection_A2A_select_all(output_layer->selection); /\* We deselect if we find an existing line *\/ */

	diff_keys = malloc(quarter * sizeof(float));
	for (line_id=0; line_id < quarter; line_id++) {
		apr_array_header_t *plottedvalues = ((apr_array_header_t **)plotted->table->elts)[line_id];
		diff_keys[line_id] = 0;
		for (j = 0; j < plottedvalues->nelts; j++) {
			float pos = ((float *)plottedvalues->elts)[j];
			if (picviz_format_is_key_axis(source->format, j+1)) {
				diff_keys[line_id] += j * pos;
			}
		}
	}

	for (line_id=quarter; line_id < plotted->table->nelts; line_id++) {
		apr_array_header_t *plottedvalues = ((apr_array_header_t **)plotted->table->elts)[line_id];
		key = 0;
		for (j = 0; j < plottedvalues->nelts; j++) {
			float pos = ((float *)plottedvalues->elts)[j];
			if (picviz_format_is_key_axis(source->format, j+1)) {
				key += j * pos;
			}
		}

		found = 0;
		for (k=0; k < quarter; k++) {
			if ((int)(key*100) == (int)(diff_keys[k]*100)) {
				found = 1;
			}
		}		
		if (!found) {
			picviz_selection_set_line(output_layer->selection, line_id, 1);
		} else {
			picviz_selection_set_line(output_layer->selection, line_id, 0);

		}
	}

	return NULL;
}

LibExport void picviz_filtering_function_terminate(void)
{

}
