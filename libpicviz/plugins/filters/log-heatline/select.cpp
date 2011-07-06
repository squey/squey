/*
 * $Id: select.cpp 2247 2011-04-08 06:42:42Z stricaud $
 * Copyright (C) Sebastien Tricaud 2010
 * Copyright (C) Philippe Saade 2010
 * 
 */

#include "core.h"

#include <picviz/general.h>
#include <picviz/debug.h>
#include <picviz/arguments.h>
#include <picviz/view.h>
#include <picviz/layer.h>
#include <picviz/selection.h>
#include <picviz/line-properties.h>
#include <picviz/filters.h>


/******************************************************************************
 *
 * picviz_filtering_function_init
 *
 *****************************************************************************/
LibCPPExport void picviz_filtering_function_init(void)
{
}



/******************************************************************************
 *
 * picviz_filtering_function_get_type
 *
 *****************************************************************************/
LibCPPExport picviz_filter_type_t picviz_filtering_function_get_type(void)
{
	return PICVIZ_FILTER_DUALSLIDER;
}



/******************************************************************************
 *
 * picviz_filtering_function_get_arguments
 *
 *****************************************************************************/
LibCPPExport picviz_arguments_t *picviz_filtering_function_get_arguments(void)
{
	picviz_arguments_t *arguments;
	picviz_argument_item_t items[] = {
	        PICVIZ_ARGUMENTS_SLIDERLEFT(Minimum, "Log Heatline Selection", 0.0),
	        PICVIZ_ARGUMENTS_SLIDERRIGHT(Maximum, "Log Heatline Selection", 1.0),
		PICVIZ_ARGUMENTS_END
		};

	arguments = picviz_arguments_new();
	picviz_arguments_item_list_append(arguments, items);

	return arguments;
}



/******************************************************************************
 *
 * picviz_filtering_function_exec
 *
 *****************************************************************************/
LibCPPExport char *picviz_filtering_function_exec(picviz_view_t *view, picviz_layer_t *input_layer, picviz_layer_t *output_layer, picviz_arguments_t *arguments)
{
	float fmin = 0;
	float fmax = 1;
	picviz_argument_item_t item;
	// Colorize is performed by an other plugin.
	//picviz_filter_plugin_heatline(view, input_layer, output_layer, picviz_filter_plugin_heatline_colorize_do, 0, 1.0);
	/* picviz_filter_plugin_heatline(view, input_layer, output_layer, _picviz_filter_plugin_heatline_select_do, fmin, fmax); */

	/* picviz_arguments_debug(arguments); */

	item = picviz_arguments_get_item_from_name(arguments, "Minimum");
	fmin = picviz_arguments_item_get_float(item);
	item = picviz_arguments_get_item_from_name(arguments, "Maximum");
	fmax = picviz_arguments_item_get_float(item);
	//picviz_arguments_debug(arguments);

//	printf(" a= %f, b=%f\n", fmin, fmax);
	picviz_filter_plugin_log_heatline(view, input_layer, output_layer, picviz_filter_plugin_log_heatline_select_do, fmin, fmax);
	return NULL;
}



/******************************************************************************
 *
 * picviz_filtering_function_terminate
 *
 *****************************************************************************/
LibCPPExport void picviz_filtering_function_terminate(void)
{

}
