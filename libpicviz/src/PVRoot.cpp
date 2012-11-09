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

/******************************************************************************
 *
 * Picviz::PVRoot::PVRoot
 *
 *****************************************************************************/
Picviz::PVRoot::PVRoot():
	data_tree_root_t(),
	_current_correlation(nullptr)
{
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
Picviz::PVAD2GView* Picviz::PVRoot::add_correlation(const QString & name)
{
	_correlations.push_back(PVAD2GView_p(new PVAD2GView(name)));
	return _correlations.back().get();
}

void Picviz::PVRoot::add_correlations(correlations_t const& corrs)
{
	for (PVAD2GView_p const& c: corrs) {
		_correlations.push_back(c);
	}
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

QList<Picviz::PVAD2GView_p> Picviz::PVRoot::get_correlations_for_scene(Picviz::PVScene const& scene) const
{
	QList<PVAD2GView_p> ret;
	for (PVAD2GView_p const& c: get_correlations()) {
		QList<Picviz::PVView*> c_views = c->get_used_views();	
		if (scene.children_belongs_to_me(c_views)) {
			ret << c;
		}
	}
	return ret;
}

void Picviz::PVRoot::serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v)
{
	_correlations.clear();
	_so_correlations = so.list("correlations", _correlations, QObject::tr("Correlations"));

	QString cur_path;
	so.attribute("current_correlation", cur_path);
	PVCore::PVSerializeObject_p so_cur_corr = _so_correlations->get_object_by_path(cur_path);
	_current_correlation = so_cur_corr->bound_obj_as<PVAD2GView>();
	PVLOG_INFO("%d correlations loaded. %p is current one.\n", _correlations.size(), _current_correlation);
	
	data_tree_root_t::serialize_read(so, v);

	_so_correlations.reset();
}

void Picviz::PVRoot::serialize_write(PVCore::PVSerializeObject& so)
{
	_so_correlations = so.list("correlations", _correlations, QObject::tr("Correlations"));
	QString cur_path = _so_correlations->get_child_path(_current_correlation);
	so.attribute("current_correlation", cur_path);

	data_tree_root_t::serialize_write(so);

	_so_correlations.reset();
}
