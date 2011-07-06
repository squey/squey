/*
 * $Id: search.cpp 2240 2011-04-06 09:59:43Z dindinx $
 * Copyright (C) Sebastien Tricaud 2010
 * Copyright (C) Philippe Saade 2010
 *
 */

#include <string>

#include <pcre.h>

#include <picviz/general.h>
#include <picviz/debug.h>
#include <picviz/view.h>
#include <picviz/layer.h>
#include <picviz/source.h>
#include <picviz/plotted.h>
#include <picviz/selection.h>
#include <picviz/arguments.h>
#include <picviz/filters.h>

#include <pvcore/general.h>

/*
 * we need to specify the vector length for our pcre_exec call.  we only care
 * about the first vector, which if the match is successful will include the
 * offset to the end of the full pattern match.  If we decide to store other
 * matches, make *SURE* that this is a multiple of 3 as pcre requires it.
 */
#define PICVIZ_SEARCH_OVECTOR_SIZE 90

LibCPPExport void picviz_filtering_function_init(void)
{
}

LibCPPExport picviz_arguments_t *picviz_filtering_function_get_arguments(void)
{
	picviz_arguments_t *arguments;
	picviz_argument_item_t items[] = {
	  PICVIZ_ARGUMENTS_TEXTBOX(Search, NULL, ""),
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
	picviz_source_t        *source;
	picviz_plotted_t       *plotted;
	picviz_argument_item_t  item;
	pcre                   *re;
	pcre_extra             *rextra = NULL;
	int                     ovector[PICVIZ_SEARCH_OVECTOR_SIZE];
	const char             *error;
	int                     erroffset;
	int                     errorcodeptr;
	std::string             str_match;
	int                     stringcount;
	int                     row = 0;
	pv_row                  nb_rows;
	int                     column = 0;
	pv_column               nb_columns;
	char                   *textbox_string;

	picviz_argument_item_t  item_axis;
	int                     axis_id;

	source = picviz_view_get_source_parent(view);
	plotted = view->PICVIZ_OBJECT_PLOTTED(parent);

	/* picviz_arguments_debug(arguments); */

	item = picviz_arguments_get_item_from_name(arguments, "Search");
	textbox_string = picviz_arguments_item_get_string(item);
	str_match = std::string(".*") + textbox_string + ".*";

	item_axis = picviz_arguments_get_item_from_name(arguments, "Axis");
	axis_id = item_axis.ival;

	re = pcre_compile2(str_match.c_str(), PCRE_CASELESS, &errorcodeptr, &error, &erroffset, NULL);
	if (!re) {
		picviz_debug(PICVIZ_DEBUG_CRITICAL, "Unable to compile regex[offset:%d]: %s.\n", erroffset, error);
		return NULL;
	}
	rextra = pcre_study(re, 0, &error);
	if (error) {
		picviz_debug(PICVIZ_DEBUG_CRITICAL, "Unable to study regex[offset:%d]: %s.\n", erroffset, error);
		return NULL;
	}


	nb_rows = picviz_nraw_count_row(source->nraw);
	nb_columns = picviz_nraw_count_columns(source->nraw);
	for (row = 0; row < nb_rows; row++) {
		for (column = 0; column < nb_columns; column++) {
			if (column == axis_id) {
				char *nraw_column_string = picviz_view_get_data(view, row, column);

				picviz_debug(PICVIZ_DEBUG_DEBUG, "Get data '%s' for row '%d' column '%d'. Comparing with '%s'\n", nraw_column_string, row, column, str_match.c_str());

				stringcount = pcre_exec(re, rextra, nraw_column_string, strlen(nraw_column_string), 0, 0, ovector, PICVIZ_SEARCH_OVECTOR_SIZE);
				if ( stringcount >= 0 ) {
					picviz_debug(PICVIZ_DEBUG_DEBUG, "********** MATCH\n");
					picviz_selection_set_line(output_layer->selection, row, 1);
					break;
				} else {
					picviz_debug(PICVIZ_DEBUG_DEBUG, "DOES NOT MATCH\n");
					picviz_selection_set_line(output_layer->selection, row, 0);
				}
			}
		}
	}

	pcre_free(re);
	if (rextra) {
		free(rextra);
	}

	return NULL;
}

LibCPPExport void picviz_filtering_function_terminate(void)
{

}
