/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <string>

#include <inendi/plugins.h>

static std::string get_inendi_plugins_path()
{
	const char* path = std::getenv("INENDI_PLUGIN_PATH");
	if (path) {
		return path;
	}
	return INENDI_PLUGIN_PATH;
}

std::string inendi_plugins_get_layer_filters_dir()
{
	return get_inendi_plugins_path() + "/layer-filters";
}

std::string inendi_plugins_get_mapping_filters_dir()
{
	return get_inendi_plugins_path() + "/mapping-filters";
}

std::string inendi_plugins_get_plotting_filters_dir()
{
	return get_inendi_plugins_path() + "/plotting-filters";
}
