//! \file filtering-function.h
//! $Id: filtering-function.h 2490 2011-04-25 02:07:58Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef _PICVIZ_FILTERING_FUNCTION_H_
#define _PICVIZ_FILTERING_FUNCTION_H_

#include <picviz/selection.h>
#include <picviz/plotted.h>
#include <picviz/view.h>

#ifdef __cplusplus
 extern "C" {
#endif



/******************************************************************************
 *
 * TYPEDEF
 *
 *****************************************************************************/

//typedef picviz_selection_t* (*picviz_filtering_heatline_function)(picviz_view_t *view, picviz_layer_t *input_layer, picviz_layer_t *output_layer, float ratio, apr_uint64_t line_id, float fmin, float fmax);
typedef void (*picviz_filtering_heatline_function)(picviz_view_t *view, picviz_layer_t *input_layer, picviz_layer_t *output_layer, float ratio, PVRow line_id, float fmin, float fmax);


#define PICVIZ_FILTERING_FUNCTION(s) picviz_filtering_function_##s

#define PICVIZ_RING_STRUCTOF(s) _##s



enum _picviz_comparison_t {
	PICVIZ_LESS,
	PICVIZ_LESS_EQUAL,
	PICVIZ_EQUAL,
	PICVIZ_GREATER,
	PICVIZ_GREATER_EQUAL,
	PICVIZ_NOT,
};
typedef enum _picviz_comparison_t picviz_comparison_t;



// LibExport picviz_selection_t *picviz_filtering_function_heatline_colorize(picviz_view_t *view, picviz_selection_t *selection);
LibExport char *picviz_filtering_function_heatline_colorize(picviz_view_t *view, picviz_layer_t *input_layer, picviz_layer_t *output_layer);
LibExport picviz_selection_t *picviz_filtering_function_less_equal(picviz_plotted_t *plotted, picviz_selection_t *selection, void *args);
LibExport char *picviz_filtering_function_square_area(picviz_view_t *view, picviz_layer_t *input_layer, picviz_layer_t *output_layer,float xmin, float ymin, float xmax, float ymax);
LibExport char *picviz_filtering_function_selectall(picviz_view_t *view, picviz_layer_t *input_layer, picviz_layer_t *output_layer);
LibExport char *picviz_filtering_function_heatline_select(picviz_view_t *view, picviz_layer_t *input_layer, picviz_layer_t *output_layer, float fmin, float fmax);



#ifdef __cplusplus
 }
#endif

#endif /* _PICVIZ_FILTERING_FUNCTION_H_ */
