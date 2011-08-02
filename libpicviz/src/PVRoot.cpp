//! \file PVRoot.cpp
//! $Id: PVRoot.cpp 3102 2011-06-10 10:43:43Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <picviz/PVRoot.h>
#include <picviz/plugins.h>

#include <pvkernel/filter/PVFilterLibrary.h>

#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/rush/PVPluginsLoad.h>

/******************************************************************************
 *
 * Picviz::PVRoot::PVRoot
 *
 *****************************************************************************/
Picviz::PVRoot::PVRoot()
{
	// PVRoot handle the filters
	load_layer_filters();
	load_mapping_filters();
	load_plotting_filters();

	// Load PVRush plugins
	PVRush::PVPluginsLoad::load_all_plugins();

	// Load PVFilter plugins
	PVFilter::PVPluginsLoad::load_all_plugins();
}

/******************************************************************************
 *
 * Picviz::PVRoot::PVRoot
 *
 *****************************************************************************/
Picviz::PVRoot::~PVRoot()
{

}

/******************************************************************************
 *
 * Picviz::PVRoot::scene_append
 *
 *****************************************************************************/
int Picviz::PVRoot::scene_append(PVScene_p scene)
{
	scenes << scene;

	return 0;
}

// Layer filters loading

/******************************************************************************
 *
 * Picviz::PVRoot::load_layer_filters
 *
 *****************************************************************************/
int Picviz::PVRoot::load_layer_filters()
{
	int ret = PVFilter::PVFilterLibraryLibLoader::load_library_from_dirs(split_plugin_dirs(QString(picviz_plugins_get_layer_filters_dir())), LAYER_FILTER_PREFIX);
	if (ret == 0) {
		PVLOG_WARN("No layer filters have been loaded !\n");
	}
	else {
		PVLOG_INFO("%d layer filters have been loaded.\n", ret);
	}
	return ret;
}

// Mapping filters loading

/******************************************************************************
 *
 * Picviz::PVRoot::load_mapping_filters
 *
 *****************************************************************************/
int Picviz::PVRoot::load_mapping_filters()
{
	int ret = PVFilter::PVFilterLibraryLibLoader::load_library_from_dirs(split_plugin_dirs(QString(picviz_plugins_get_mapping_filters_dir())), MAPPING_FILTER_PREFIX);
	if (ret == 0) {
		PVLOG_WARN("No mapping filters have been loaded !\n");
	}
	else {
		PVLOG_INFO("%d mapping filters have been loaded.\n", ret);
	}
	return ret;
}

// Plotting filters loading

/******************************************************************************
 *
 * Picviz::PVRoot::load_plotting_filters
 *
 *****************************************************************************/
int Picviz::PVRoot::load_plotting_filters()
{
	int ret = PVFilter::PVFilterLibraryLibLoader::load_library_from_dirs(split_plugin_dirs(QString(picviz_plugins_get_plotting_filters_dir())), PLOTTING_FILTER_PREFIX);
	if (ret == 0) {
		PVLOG_WARN("No plotting filters have been loaded !\n");
	}
	else {
		PVLOG_INFO("%d plotting filters have been loaded.\n", ret);
	}
	return ret;
}

QStringList Picviz::PVRoot::split_plugin_dirs(QString const& dirs)
{
	return dirs.split(PVCORE_DIRECTORY_SEP);
}
