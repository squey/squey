//! \file filters.h
//! $Id: filters.h 2490 2011-04-25 02:07:58Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef _PICVIZ_FILTERS_H_
#define _PICVIZ_FILTERS_H_

#include <apr_general.h>
#include <apr_hash.h>
#include <apr_dso.h>

#include <picviz/general.h>
#include <picviz/layer.h>
#include <picviz/PVView.h>
#include <picviz/arguments.h>
#include <picviz/datatreerootitem.h>

/**
 * \defgroup PicvizFunctions Picviz Functions
 * @{
 */

#ifdef __cplusplus
 extern "C" {
#endif

typedef void (*picviz_filtering_function_init_func)(void);
typedef picviz_arguments_t *(*picviz_filtering_function_get_arguments_func)(void);
typedef char* (*picviz_filtering_function_exec_func)(Picviz::PVView *view, picviz_layer_t *input_layer, picviz_layer_t *output_layer, picviz_arguments_t *arguments);
typedef void (*picviz_filtering_function_terminate_func)(void);

#ifndef picviz_filtering_function_init_string
    #define picviz_filtering_function_init_string "picviz_filtering_function_init"
#endif
#ifndef picviz_filtering_function_get_type_string
    #define picviz_filtering_function_get_type_string "picviz_filtering_function_get_type"
#endif
#ifndef picviz_filtering_function_get_arguments_string
    #define picviz_filtering_function_get_arguments_string "picviz_filtering_function_get_arguments"
#endif
#ifndef picviz_filtering_function_exec_string
    #define picviz_filtering_function_exec_string "picviz_filtering_function_exec"
#endif
#ifndef picviz_filtering_function_terminate_string
    #define picviz_filtering_function_terminate_string "picviz_filtering_function_terminate"
#endif

enum _picviz_filter_type_t {
	PICVIZ_FILTER_NOCONFIG, /* This is what we have when we do not have any arguments */
	PICVIZ_FILTER_NOWIDGET, /* When we cannot use any widget, each argument is put in a box that the user can set */
	PICVIZ_FILTER_GUIWIDGET, /* The widget is defined in the GUI and knows how to deal with this specific plugin */
	PICVIZ_FILTER_SINGLESLIDER,
	PICVIZ_FILTER_DUALSLIDER,
};
typedef enum _picviz_filter_type_t picviz_filter_type_t;
typedef picviz_filter_type_t (*picviz_filtering_function_get_type_func)(void);

struct _picviz_filter_t {
	apr_dso_handle_t *dso;

	picviz_filter_type_t type;

	picviz_arguments_t *arguments;

	picviz_filtering_function_init_func init_func;
	picviz_filtering_function_get_arguments_func get_arguments_func;
	picviz_filtering_function_get_type_func get_type_func;
	picviz_filtering_function_exec_func exec_func;
	picviz_filtering_function_terminate_func terminate_func;
};
typedef struct _picviz_filter_t picviz_filter_t;

typedef void (*picviz_filtering_function_foreach_func)(char *name, picviz_filter_t *filter, void *userdata);

LibExport picviz_filter_t *picviz_filter_new(void);
LibExport picviz_filter_t *picviz_filter_plugin_load(apr_pool_t *pool, char *filepath);
LibExport int picviz_filters_plugin_register_all(apr_pool_t *pool, apr_hash_t *hash);
LibExport void picviz_filters_foreach_filter(picviz_datatreerootitem_t *datatree, picviz_filtering_function_foreach_func foreach_func, void *userdata);
LibExport picviz_filter_t *picviz_filters_get_filter_from_name(picviz_datatreerootitem_t *datatree, char *name);

#ifdef __cplusplus
 }
#endif

/*@}*/

#endif /* _PICVIZ_FILTERS_H_ */


