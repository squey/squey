/*
 * $Id$
 * Copyright (C) Sebastien Tricaud 2010
 * Copyright (C) Philippe Saade 2010
 * 
 */

#include <stdlib.h>
#include <string.h>

#include <pcre.h>

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
#include <picviz/file.h>
#include <picviz/plugins.h>


#define BLACKLIST_TOULOUSE_FILE "blacklist-toulouse.txt"
/* UT_array *blacklist_toulouse_array; */
char *blacklist_regex[500000];
int blacklist_regex_pos = 0;

picviz_file_t *blacklist_file;


/*
 * we need to specify the vector length for our pcre_exec call.  we only care
 * about the first vector, which if the match is successful will include the
 * offset to the end of the full pattern match.  If we decide to store other
 * matches, make *SURE* that this is a multiple of 3 as pcre requires it.
 */
#define PICVIZ_BLACKLIST_OVECTOR_SIZE 90


int blacklist_toulouse_line_foreach(long offset, char *line, size_t linesize, void *userdata) 
{
	char *match_string;
	const char *error;
	int erroffset;
	int errorcodeptr;
	size_t len;

	len = strlen("http:\\/\\/") + strlen((char *)line) + 1;
	blacklist_regex[blacklist_regex_pos] = malloc(len);
	sprintf(blacklist_regex[blacklist_regex_pos], "http:\\/\\/%s", (char *)line, len);
	blacklist_regex_pos++;

	return 0;
}

LibExport void picviz_filtering_function_init(void) 
{
	char *path_to_blacklist;
	size_t path_to_blacklist_len;

	/* utarray_new(blacklist_toulouse_array, &ut_str_icd); */

	path_to_blacklist_len = strlen(picviz_plugins_get_filters_dir());
	path_to_blacklist_len += strlen(BLACKLIST_TOULOUSE_FILE);
	path_to_blacklist_len += 2; /* 2 = '/' + '\0' */

	path_to_blacklist = malloc(path_to_blacklist_len);
#ifdef WIN32
	sprintf(path_to_blacklist, "%s%c%s", picviz_plugins_get_filters_dir(), PICVIZ_PATH_SEPARATOR_CHAR, BLACKLIST_TOULOUSE_FILE);
#else
	snprintf(path_to_blacklist, path_to_blacklist_len, "%s%c%s", picviz_plugins_get_filters_dir(), PICVIZ_PATH_SEPARATOR_CHAR, BLACKLIST_TOULOUSE_FILE);
#endif

	blacklist_file = picviz_file_new(path_to_blacklist);

	picviz_file_line_foreach(blacklist_file, blacklist_toulouse_line_foreach, NULL);

	free(path_to_blacklist);
}

LibExport picviz_arguments_t *picviz_filtering_function_get_arguments(void)
{
	picviz_arguments_t *arguments;
	picviz_argument_item_t items[] = {
	  PICVIZ_ARGUMENTS_TEXTBOX(Axis, NULL, "")
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
	picviz_source_t *source;
	picviz_plotted_t *plotted;
	picviz_argument_item_t item;

	int line_id;

	pcre *re;
        pcre_extra *rextra = NULL;
	const char *error;
	int erroffset;
	int errorcodeptr;
	char *str_match;
	int stringcount;

	int row = 0;
	pv_row nb_rows;
	int column = 0;
	pv_column nb_columns;
	char *textbox_string;

	picviz_argument_item_t item_axis;
	char *axis_string;
	int axis_id;
	
	int i;
	int nblines;
	int ret;

	source = picviz_view_get_source_parent(view);
	plotted = view->PICVIZ_OBJECT_PLOTTED(parent);


	item_axis = picviz_arguments_get_item_from_name(arguments, "Axis");
	axis_string = picviz_arguments_item_get_string(item_axis);
	axis_id = atoi(axis_string);


	picviz_selection_A2A_select_none(output_layer->selection);

	nblines = blacklist_file->nblines;
	picviz_file_destroy(blacklist_file);

	nb_rows = picviz_view_get_row_count(view);
	for (row = 0; row < nb_rows; row++) {
		char *nraw_string = picviz_view_get_data(view, row, axis_id);

		for (i=0; i < nblines; i++) {
			ret = strncmp(nraw_string, blacklist_regex[i], strlen(blacklist_regex[i]));
			if (!ret) {
				picviz_selection_set_line(output_layer->selection, row, 1);
			}
			/* stringcount = pcre_exec(blacklist_regex[i], NULL, nraw_string, strlen(nraw_string), 0, 0, ovector, PICVIZ_BLACKLIST_OVECTOR_SIZE); */
			/* if (stringcount >= 0) { */
			/* 	picviz_selection_set_line(output_layer->selection, row, 1); */
			/* } */
		}
	}

	return NULL;
}

LibExport void picviz_filtering_function_terminate(void)
{

}
