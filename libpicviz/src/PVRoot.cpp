/**
 * \file PVRoot.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <picviz/PVRoot.h>
#include <picviz/PVScene.h>
#include <picviz/plugins.h>

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/rush/PVPluginsLoad.h>

#include <tulip/TlpTools.h>

Picviz::PVRoot_sp Picviz::PVRoot::_unique_root;

/******************************************************************************
 *
 * Picviz::PVRoot::PVRoot
 *
 *****************************************************************************/
Picviz::PVRoot::PVRoot() : data_tree_root_t()
{
	// Tulip initialisation
	tlp::initTulipLib();

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
 * Picviz::PVRoot::PVRoot
 *
 *****************************************************************************/
Picviz::PVRoot::~PVRoot()
{
	PVLOG_INFO("In PVRoot destructor\n");
}

Picviz::PVRoot& Picviz::PVRoot::get_root()
{
	return *get_root_sp();
}

Picviz::PVRoot_sp Picviz::PVRoot::get_root_sp()
{
	if (!_unique_root) {
		_unique_root.reset(new Picviz::PVRoot());
	}
	return _unique_root;
}

void Picviz::PVRoot::release()
{
	_unique_root.reset();
}


/******************************************************************************
 *
 * Picviz::PVRoot::add_correlation
 *
 *****************************************************************************/
void Picviz::PVRoot::add_correlation()
{
	_correlations.push_back(PVAD2GView_p(new PVAD2GView()));
}

/******************************************************************************
 *
 * Picviz::PVRoot::delete_correlation
 *
 *****************************************************************************/
void Picviz::PVRoot::delete_correlation(int index)
{
	correlations_t::iterator i = _correlations.begin();
	std::advance(i, index);
	_correlations.erase(i);
}

/******************************************************************************
 *
 * Picviz::PVRoot::load_axis_computation_filters
 *
 *****************************************************************************/
int Picviz::PVRoot::load_axis_computation_filters()
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
int Picviz::PVRoot::load_layer_filters()
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
int Picviz::PVRoot::load_mapping_filters()
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
int Picviz::PVRoot::load_plotting_filters()
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
int Picviz::PVRoot::load_row_filters()
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
int Picviz::PVRoot::load_sorting_functions_filters()
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
