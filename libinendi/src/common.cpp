/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/rush/PVPluginsLoad.h>

#include <inendi/common.h>
#include <inendi/plugins.h>
#include <inendi/PVRoot.h>
#include <inendi/PVScene.h>
#include <inendi/PVView.h>

void Inendi::common::load_filters()
{
	// PVRoot handle the filters
	load_layer_filters();
	load_mapping_filters();
	load_plotting_filters();
	load_axis_computation_filters();
	load_sorting_functions_filters();

	// Load PVRush plugins
	PVRush::PVPluginsLoad::load_all_plugins();

	// Load PVFilter plugins
	PVFilter::PVPluginsLoad::load_all_plugins();
}

/******************************************************************************
 *
 * Inendi::PVRoot::load_axis_computation_filters
 *
 *****************************************************************************/
int Inendi::common::load_axis_computation_filters()
{
	int ret = PVCore::PVClassLibraryLibLoader::load_class_from_dirs(QString::fromStdString(inendi_plugins_get_axis_computation_dir()), AXIS_COMPUTATION_PLUGINS_PREFIX);
	if (ret == 0) {
		PVLOG_WARN("No axis computation plugin has been loaded !\n");
	}
	else {
		PVLOG_INFO("%d axis computation plugins have been loaded.\n", ret);
	}
	return ret;
}

// Layer filters loading

/******************************************************************************
 *
 * Inendi::PVRoot::load_layer_filters
 *
 *****************************************************************************/
int Inendi::common::load_layer_filters()
{
	int ret = PVCore::PVClassLibraryLibLoader::load_class_from_dirs(QString::fromStdString(inendi_plugins_get_layer_filters_dir()), LAYER_FILTER_PREFIX);
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
 * Inendi::PVRoot::load_mapping_filters
 *
 *****************************************************************************/
int Inendi::common::load_mapping_filters()
{
	int ret = PVCore::PVClassLibraryLibLoader::load_class_from_dirs(QString::fromStdString(inendi_plugins_get_mapping_filters_dir()), MAPPING_FILTER_PREFIX);
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
 * Inendi::PVRoot::load_plotting_filters
 *
 *****************************************************************************/
int Inendi::common::load_plotting_filters()
{
	int ret = PVCore::PVClassLibraryLibLoader::load_class_from_dirs(QString::fromStdString(inendi_plugins_get_plotting_filters_dir()), PLOTTING_FILTER_PREFIX);
	if (ret == 0) {
		PVLOG_WARN("No plotting filters have been loaded !\n");
	}
	else {
		PVLOG_INFO("%d plotting filters have been loaded.\n", ret);
	}
	return ret;
}

/******************************************************************************
 *
 * Inendi::PVRoot::load_sorting_functions_filters
 *
 *****************************************************************************/
int Inendi::common::load_sorting_functions_filters()
{
	int ret = PVCore::PVClassLibraryLibLoader::load_class_from_dirs(QString::fromStdString(inendi_plugins_get_sorting_functions_dir()), SORTING_FUNCTIONS_PLUGINS_PREFIX);
	if (ret == 0) {
		PVLOG_WARN("No sorting plugin has been loaded !\n");
	}
	else {
		PVLOG_INFO("%d sorting plugins have been loaded.\n", ret);
	}
	return ret;
}
