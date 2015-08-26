#include <pvkernel/core/general.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/rush/PVPluginsLoad.h>

#include <picviz/common.h>
#include <picviz/plugins.h>
#include <picviz/PVRoot.h>
#include <picviz/PVScene.h>
#include <picviz/PVView.h>

#ifdef ENABLE_CORRELATION
#include <tulip/TlpTools.h>
#endif

void Picviz::common::load_filters()
{
#ifdef ENABLE_CORRELATION
	// Tulip initialisation
	tlp::initTulipLib();
#endif

	// PVRoot handle the filters
	load_layer_filters();
	load_mapping_filters();
	load_plotting_filters();
	load_row_filters();
	load_axis_computation_filters();
	load_sorting_functions_filters();

	// Load PVRush plugins
	PVRush::PVPluginsLoad::load_all_plugins();

	// Load PVFilter plugins
	PVFilter::PVPluginsLoad::load_all_plugins();
}

/******************************************************************************
 *
 * Picviz::PVRoot::load_axis_computation_filters
 *
 *****************************************************************************/
int Picviz::common::load_axis_computation_filters()
{
	int ret = PVCore::PVClassLibraryLibLoader::load_class_from_dirs(QString(picviz_plugins_get_axis_computation_dir()), AXIS_COMPUTATION_PLUGINS_PREFIX);
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
 * Picviz::PVRoot::load_layer_filters
 *
 *****************************************************************************/
int Picviz::common::load_layer_filters()
{
	int ret = PVCore::PVClassLibraryLibLoader::load_class_from_dirs(QString(picviz_plugins_get_layer_filters_dir()), LAYER_FILTER_PREFIX);
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
int Picviz::common::load_mapping_filters()
{
	int ret = PVCore::PVClassLibraryLibLoader::load_class_from_dirs(QString(picviz_plugins_get_mapping_filters_dir()), MAPPING_FILTER_PREFIX);
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
int Picviz::common::load_plotting_filters()
{
	int ret = PVCore::PVClassLibraryLibLoader::load_class_from_dirs(QString(picviz_plugins_get_plotting_filters_dir()), PLOTTING_FILTER_PREFIX);
	if (ret == 0) {
		PVLOG_WARN("No plotting filters have been loaded !\n");
	}
	else {
		PVLOG_INFO("%d plotting filters have been loaded.\n", ret);
	}
	return ret;
}


// Row filters loading

/******************************************************************************
 *
 * Picviz::PVRoot::load_row_filters
 *
 *****************************************************************************/
int Picviz::common::load_row_filters()
{
	int ret = PVCore::PVClassLibraryLibLoader::load_class_from_dirs(QString(picviz_plugins_get_row_filters_dir()), ROW_FILTER_PREFIX);
	if (ret == 0) {
		PVLOG_WARN("No row filters have been loaded !\n");
	}
	else {
		PVLOG_INFO("%d row filters have been loaded.\n", ret);
	}
	return ret;
}

/******************************************************************************
 *
 * Picviz::PVRoot::load_sorting_functions_filters
 *
 *****************************************************************************/
int Picviz::common::load_sorting_functions_filters()
{
	int ret = PVCore::PVClassLibraryLibLoader::load_class_from_dirs(QString(picviz_plugins_get_sorting_functions_dir()), SORTING_FUNCTIONS_PLUGINS_PREFIX);
	if (ret == 0) {
		PVLOG_WARN("No sorting plugin has been loaded !\n");
	}
	else {
		PVLOG_INFO("%d sorting plugins have been loaded.\n", ret);
	}
	return ret;
}
