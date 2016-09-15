/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/rush/PVPluginsLoad.h>

#include <pvkernel/filter/PVPluginsLoad.h>

#include <pvkernel/core/PVLogger.h>
#include <pvkernel/core/PVClassLibrary.h>

#include <inendi/common.h>
#include <inendi/plugins.h>

void Inendi::common::load_filters()
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

// Layer filters loading

/******************************************************************************
 *
 * Inendi::common::load_layer_filters
 *
 *****************************************************************************/
int Inendi::common::load_layer_filters()
{
	int ret = PVCore::PVClassLibraryLibLoader::load_class_from_dirs(
	    QString::fromStdString(inendi_plugins_get_layer_filters_dir()), "layer_filter");
	if (ret == 0) {
		PVLOG_WARN("No layer filters have been loaded !\n");
	} else {
		PVLOG_INFO("%d layer filters have been loaded.\n", ret);
	}

	return ret;
}

// Mapping filters loading

/******************************************************************************
 *
 * Inendi::common::load_mapping_filters
 *
 *****************************************************************************/
int Inendi::common::load_mapping_filters()
{
	int ret = PVCore::PVClassLibraryLibLoader::load_class_from_dirs(
	    QString::fromStdString(inendi_plugins_get_mapping_filters_dir()), "mapping_filter");
	if (ret == 0) {
		PVLOG_WARN("No mapping filters have been loaded !\n");
	} else {
		PVLOG_INFO("%d mapping filters have been loaded.\n", ret);
	}
	return ret;
}

// Plotting filters loading

/******************************************************************************
 *
 * Inendi::common::load_plotting_filters
 *
 *****************************************************************************/
int Inendi::common::load_plotting_filters()
{
	int ret = PVCore::PVClassLibraryLibLoader::load_class_from_dirs(
	    QString::fromStdString(inendi_plugins_get_plotting_filters_dir()), "plotting_filter");
	if (ret == 0) {
		PVLOG_WARN("No plotting filters have been loaded !\n");
	} else {
		PVLOG_INFO("%d plotting filters have been loaded.\n", ret);
	}
	return ret;
}
