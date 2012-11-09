/**
 * \file PVRoot.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <picviz/PVRoot.h>
#include <picviz/PVScene.h>
#include <picviz/PVView.h>
#include <picviz/plugins.h>

#include <tulip/TlpTools.h>

Picviz::PVRoot_sp Picviz::PVRoot::_unique_root;

/******************************************************************************
 *
 * Picviz::PVRoot::PVRoot
 *
 *****************************************************************************/
Picviz::PVRoot::PVRoot() : data_tree_root_t()
{

	_available_colors << 0x9966CC << 0x6699CC << 0x778800 << 0xFFCC66 << 0x993366 << 0x999999 << 0x339999 << 0xFF6633 << 0x99FFCC << 0xFFFF99;
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

Picviz::PVView* Picviz::PVRoot::current_view()
{
	return _current_view;
}

Picviz::PVView const* Picviz::PVRoot::current_view() const
{
	return const_cast<PVView const*>(const_cast<PVRoot*>(this)->current_view());
}

void Picviz::PVRoot::select_view(PVView& view)
{
	 _current_view = &view;
}


/******************************************************************************
 *
 * Picviz::PVRoot::get_new_view_id
 *
 *****************************************************************************/
Picviz::PVView::id_t Picviz::PVRoot::get_new_view_id()
{
	return _new_view_id++;
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
QColor Picviz::PVRoot::get_new_view_color()
{
	if (_available_colors.size() == 0) {
		std::swap(_available_colors, _used_colors);
	}
	QRgb color = _available_colors.at(0);
	_available_colors.pop_front();
	_used_colors << color;
	return color;
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
