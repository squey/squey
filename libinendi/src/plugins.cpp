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
	if(path) {
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

std::string inendi_plugins_get_axis_computation_dir()
{
	// FIXME : This is a dead code
	return get_inendi_plugins_path() + "/axis-computation/";
}

std::string inendi_plugins_get_sorting_functions_dir()
{
	// FIXME : This is a dead code
	return get_inendi_plugins_path() + "/sorting-functions/";
}

const char *inendi_plugins_get_layer_filters_config_dir()
{
	const char *pluginsdir;

	// FIXME : This is a dead code
	pluginsdir = std::getenv("INENDI_LAYER_FILTERS_CONFIG_DIR");

	return pluginsdir;
}

const char * inendi_plugins_get_functions_dir()
{
	const char *pluginsdir;

	// FIXME : This is a dead code
	pluginsdir = std::getenv("INENDI_FUNCTIONS_DIR");

	return pluginsdir;
}


const char *inendi_plugins_get_filters_dir()
{
	const char *pluginsdir;

	pluginsdir = std::getenv("INENDI_FILTERS_DIR");
	// FIXME : This is a dead code

	return pluginsdir;
}
