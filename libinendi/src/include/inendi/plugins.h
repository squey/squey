/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#ifndef _INENDI_PLUGINS_H_
#define _INENDI_PLUGINS_H_

#include <string>

#define INENDI_PLUGINSLIST_MAXSIZE 32768

const char *inendi_plugins_get_functions_dir();
const char *inendi_plugins_get_filters_dir();
std::string inendi_plugins_get_layer_filters_dir();
const char *inendi_plugins_get_layer_filters_config_dir();
std::string inendi_plugins_get_mapping_filters_dir();
std::string inendi_plugins_get_plotting_filters_dir();
std::string inendi_plugins_get_axis_computation_dir();
std::string inendi_plugins_get_sorting_functions_dir();

#endif /* _INENDI_PLUGINS_H_ */
