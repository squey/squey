/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef _INENDI_PLUGINS_H_
#define _INENDI_PLUGINS_H_

#include <inendi/general.h>
#include <QString>

#define INENDI_PLUGINSLIST_MAXSIZE 32768

const char *inendi_plugins_get_functions_dir(void);
const char *inendi_plugins_get_filters_dir(void);
QString inendi_plugins_get_layer_filters_dir(void);
const char *inendi_plugins_get_layer_filters_config_dir(void);
QString inendi_plugins_get_mapping_filters_dir(void);
QString inendi_plugins_get_plotting_filters_dir(void);
QString inendi_plugins_get_axis_computation_dir(void);
QString inendi_plugins_get_sorting_functions_dir(void);

#endif /* _INENDI_PLUGINS_H_ */
