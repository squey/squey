/*
 * $Id: colorize.cpp 2255 2011-04-08 08:58:04Z stricaud $
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


LibCPPExport void picviz_filtering_function_init(void)
{
}


LibCPPExport picviz_filter_type_t picviz_filtering_function_get_type(void)
{
	return PICVIZ_FILTER_NOCONFIG;
}

LibCPPExport picviz_arguments_t *picviz_filtering_function_get_arguments(void)
{
	return NULL;
}

LibCPPExport picviz_selection_t *picviz_filtering_function_exec(picviz_view_t *view, picviz_layer_t *input_layer, picviz_layer_t *output_layer, picviz_arguments_t *arguments)
{
  picviz_filter_plugin_log_heatline(view, input_layer, output_layer, picviz_filter_plugin_log_heatline_colorize_do, 0, 1.0);
	return NULL;
}

LibCPPExport void picviz_filtering_function_terminate(void)
{

}

