/**
 * \file PVRoot.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <picviz/PVRoot.h>
#include <picviz/PVScene.h>
#include <picviz/PVView.h>
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
 * Picviz::PVRoot::get_new_view_id
 *
 *****************************************************************************/
Picviz::PVView::id_t Picviz::PVRoot::get_new_view_id() const
{
	return get_children<PVView>().size()-1;
}

/******************************************************************************
 *
 * Picviz::PVRoot::set_views_id
 *
 *****************************************************************************/
void Picviz::PVRoot::set_views_id()
{
	std::multimap<PVView::id_t, PVView*> map_views;
	for (auto view : get_children<PVView>()) {
		map_views.insert(std::make_pair(view->get_view_id(), view.get()));
	}
	PVView::id_t cur_id = 0;
	std::multimap<PVView::id_t, PVView*>::iterator it;
	for (it = map_views.begin(); it != map_views.end(); it++) {
		it->second->set_view_id(cur_id);
		cur_id++;
	}
}

/******************************************************************************
 *
 * Picviz::PVRoot::get_new_view_color
 *
 *****************************************************************************/
QColor Picviz::PVRoot::get_new_view_color() const
{
	return QColor(_view_colors[(get_new_view_id()) % (sizeof(_view_colors)/sizeof(QRgb))]);
}

/******************************************************************************
 *
 * Picviz::PVRoot::get_correlation
 *
 *****************************************************************************/
Picviz::PVAD2GView_p Picviz::PVRoot::get_correlation(int index)
{
	correlations_t::iterator i = _correlations.begin();
	std::advance(i, index);
	return *i;
}

/******************************************************************************
 *
 * Picviz::PVRoot::add_correlation
 *
 *****************************************************************************/
void Picviz::PVRoot::add_correlation(const QString & name)
{
	_correlations.push_back(PVAD2GView_p(new PVAD2GView(name)));
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
 * Picviz::PVRoot::remove_view_from_correlations
 *
 *****************************************************************************/
void Picviz::PVRoot::remove_view_from_correlations(PVView* view)
{
	for (PVAD2GView_p correlation : _correlations) {
		correlation->del_view(view);
	}
}

/******************************************************************************
 *
 * Picviz::PVRoot::process_correlation
 *
 *****************************************************************************/
QList<Picviz::PVView*> Picviz::PVRoot::process_correlation(PVView* src_view)
{
	QList<Picviz::PVView*> changed_views;
	if (_correlations_enabled && _current_correlation && !_correlation_running) {
		_correlation_running = true;
		_current_correlation->pre_process();
		_current_correlation->run(src_view, &changed_views);
		for (Picviz::PVView* view : changed_views) {
			view->process_from_selection();
		}
		_correlation_running = false;
	}

	return changed_views;
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
