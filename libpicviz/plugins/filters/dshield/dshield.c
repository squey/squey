/*
 * $Id$
 * Copyright (C) Sebastien Tricaud 2010
 * Copyright (C) Philippe Saade 2010
 * 
 */

#include <stdio.h>
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
#include <picviz/plugins.h>
#include <picviz/file.h>

#include <uthash/utarray.h>

#define DSHIELD_IPLIST_FILE "dshield-ipascii.txt"
UT_array *dshield_iplist_array;


LibExport void picviz_filtering_function_init(void)
{
}

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

int dshield_iplist_line_foreach(long offset, char *line, size_t linesize, void *userdata) 
{
  utarray_push_back(dshield_iplist_array, &line);
  return 0;
}

LibExport char *picviz_filtering_function_exec(picviz_view_t *view, picviz_layer_t *input_layer, picviz_layer_t *output_layer, picviz_arguments_t *arguments)
{

	char *path_to_iplist;
	size_t path_to_iplist_len;

	picviz_plotted_t *plotted;
	picviz_argument_item_t item_axis;
	picviz_argument_item_t item_value;

	int line_id;

	int row = 0;
	pv_row nb_rows;
	int column = 0;
	pv_column nb_columns;

	char *value_string;

	int axis_id;
	float value;

	picviz_file_t *file;
	char **str_p;

	plotted = view->PICVIZ_OBJECT_PLOTTED(parent);

	item_axis = picviz_arguments_get_item_from_name(arguments, "Axis");
	axis_id = item_axis.ival;

	/* We build the array with all values from the dshield normalized file */
	utarray_new(dshield_iplist_array, &ut_str_icd);

	path_to_iplist_len = strlen(picviz_plugins_get_filters_dir());
	path_to_iplist_len += strlen(DSHIELD_IPLIST_FILE);
	path_to_iplist_len += 2; /* 2 = '/' + '\0' */

	path_to_iplist = malloc(path_to_iplist_len);
#ifdef WIN32
	sprintf(path_to_iplist, "%s%c%s", picviz_plugins_get_filters_dir(), PICVIZ_PATH_SEPARATOR_CHAR, DSHIELD_IPLIST_FILE);
#else
	snprintf(path_to_iplist, path_to_iplist_len, "%s%c%s", picviz_plugins_get_filters_dir(), PICVIZ_PATH_SEPARATOR_CHAR, DSHIELD_IPLIST_FILE);
#endif

	file = picviz_file_new(path_to_iplist);
	picviz_file_line_foreach(file, dshield_iplist_line_foreach, NULL);
	picviz_file_destroy(file);

	free(path_to_iplist);


	picviz_selection_A2A_select_none(output_layer->selection);
	nb_rows = picviz_view_get_row_count(view);
	for (row = 0; row < nb_rows; row++) {
		char *nraw_string = picviz_view_get_data(view, row, axis_id);

		str_p = NULL;
		while ( (str_p=(char**)utarray_next(dshield_iplist_array,str_p))) {
		  if (!strcmp(nraw_string, *str_p)) {
		        picviz_selection_set_line(output_layer->selection, row, 1);
		  }
		}
	}



	utarray_free(dshield_iplist_array);

	return NULL;
}

LibExport void picviz_filtering_function_terminate(void)
{

}
