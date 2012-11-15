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

#define ARCHIVE_ROOT_DESC (QObject::tr("Solution"))

/******************************************************************************
 *
 * Picviz::PVRoot::PVRoot
 *
 *****************************************************************************/
Picviz::PVRoot::PVRoot():
	data_tree_root_t(),
	_current_scene(nullptr),
	_current_source(nullptr),
	_current_view(nullptr),
	_current_correlation(nullptr)
{
	reset_colors();
}

/******************************************************************************
 *
 * Picviz::PVRoot::PVRoot
 *
 *****************************************************************************/
Picviz::PVRoot::~PVRoot()
{
	remove_all_children();
	PVLOG_INFO("In PVRoot destructor\n");
}

void Picviz::PVRoot::clear()
{
	remove_all_children();
	_current_scene = nullptr;
	_current_source = nullptr;
	_current_view = nullptr;
	_current_correlation = nullptr;
	_correlations.clear();
	_correlation_running = false;
	_correlations_enabled = true;
	_so_correlations.reset();
	_original_archive.reset();
	_path.clear();
	_new_view_id = 0;
	reset_colors();
}

void Picviz::PVRoot::reset_colors()
{
	_available_colors.clear();
	_available_colors << 0x9966CC << 0x6699CC << 0x778800 << 0xFFCC66 << 0x993366 << 0x999999 << 0x339999 << 0xFF6633 << 0x99FFCC << 0xFFFF99;
	_used_colors.clear();
}

void Picviz::PVRoot::select_view(PVView& view)
{
	assert(view.get_parent<PVRoot>() == this);
	_current_view = &view;
	_current_scene = view.get_parent<PVScene>();
	_current_source = view.get_parent<PVSource>();

	_current_scene->set_last_active_source(_current_source);
	_current_source->set_last_active_view(&view);
}

void Picviz::PVRoot::select_source(PVSource& src)
{
	assert(src.get_parent<PVRoot>() == this);
	_current_source = &src;
	_current_view = src.last_active_view();
	_current_scene = src.get_parent<PVScene>();

	_current_scene->set_last_active_source(&src);
}

void Picviz::PVRoot::select_scene(PVScene& scene)
{
	assert(scene.get_parent<PVRoot>() == this);
	_current_scene = &scene;
	_current_source = scene.last_active_source();
	if (_current_source) {
		_current_view = _current_source->last_active_view();
	}
}

void Picviz::PVRoot::view_being_deleted(Picviz::PVView* view)
{
	if (_current_view == view) {
		_current_view = nullptr;
	}
}

void Picviz::PVRoot::scene_being_deleted(Picviz::PVScene* scene)
{
	if (_current_scene == scene) {
		_current_scene = nullptr;
	}
	if (_current_source && _current_source->get_parent<PVScene>() == scene) {
		_current_source = nullptr;
	}
	if (_current_view && _current_view->get_parent<PVScene>() == scene) {
		_current_view = nullptr;
	}
}

void Picviz::PVRoot::source_being_deleted(Picviz::PVSource* src)
{
	if (_current_source == src) {
		_current_source = nullptr;
	}
	if (_current_view && _current_view->get_parent<PVSource>() == src) {
		_current_view = nullptr;
	}
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
Picviz::PVAD2GView* Picviz::PVRoot::add_correlation(const QString & name)
{
	Picviz::PVAD2GView* correlation = new Picviz::PVAD2GView(name);
	_correlations.push_back(PVAD2GView_p(correlation));

	return correlation;
}


/******************************************************************************
 *
 * Picviz::PVRoot::add_correlations
 *
 *****************************************************************************/
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
bool Picviz::PVRoot::delete_correlation(PVAD2GView_p correlation_p)
{
	for (auto it=_correlations.begin() ; it != _correlations.end(); it++) {
		if ((*it).get() == correlation_p.get()) {
			_correlations.erase(it);
			return true;
		}
	}

	return false;
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

Picviz::PVRoot::correlations_t Picviz::PVRoot::get_correlations_for_scene(Picviz::PVScene const& scene) const
{
	Picviz::PVRoot::correlations_t ret;
	for (PVAD2GView_p const& c: get_correlations()) {
		QList<Picviz::PVView*> c_views = c->get_used_views();
		if (scene.children_belongs_to_me(c_views)) {
			ret.push_back(c);
		}
	}
	return ret;
}

Picviz::PVScene* Picviz::PVRoot::get_scene_from_path(const QString& path)
{
	for (Picviz::PVScene_sp const& scene: get_children()) {
		if (scene->get_path() == path) {
			return scene.get();
		}
	}
	return nullptr;
}

void Picviz::PVRoot::serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v)
{
	_correlations.clear();

	data_tree_root_t::serialize_read(so, v);

	_so_correlations = so.list("correlations", _correlations, QObject::tr("Correlations"), (PVAD2GView*) NULL, QStringList(), true, true);
	if (_so_correlations) {
		QString cur_path;
		so.attribute("current_correlation", cur_path);
		PVCore::PVSerializeObject_p so_cur_corr = _so_correlations->get_object_by_path(cur_path);
		if (so_cur_corr) {
			_current_correlation = so_cur_corr->bound_obj_as<PVAD2GView>();
		}
		else {
			_current_correlation = nullptr;
		}
		PVLOG_INFO("%d correlations loaded. %p is current one.\n", _correlations.size(), _current_correlation);
	}

	_so_correlations.reset();
}

void Picviz::PVRoot::serialize_write(PVCore::PVSerializeObject& so)
{
	QStringList corr_desc;
	corr_desc.reserve(_correlations.size());
	for (PVAD2GView_p const& c: _correlations) {
		corr_desc << c->get_name();
	}

	_so_correlations = so.list("correlations", _correlations, QObject::tr("Correlations"), (Picviz::PVAD2GView*) NULL, corr_desc, true, true);
	QString cur_path = _so_correlations->get_child_path(_current_correlation);
	so.attribute("current_correlation", cur_path);

	data_tree_root_t::serialize_write(so);

	_so_correlations.reset();
}

void Picviz::PVRoot::save_to_file(QString const& path, PVCore::PVSerializeArchiveOptions_p options, bool save_everything)
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	set_path(path);
	PVCore::PVSerializeArchive_p ar(new PVCore::PVSerializeArchiveZip(path, PVCore::PVSerializeArchive::write, PICVIZ_ARCHIVES_VERSION));
	if (options) {
		ar->set_options(options);
	}
	ar->set_save_everything(save_everything);
	ar->get_root()->object("root", *this, ARCHIVE_ROOT_DESC);
	ar->finish();
#endif
}

void Picviz::PVRoot::load_from_file(QString const& path)
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	PVCore::PVSerializeArchive_p ar(new PVCore::PVSerializeArchiveZip(path, PVCore::PVSerializeArchive::read, PICVIZ_ARCHIVES_VERSION));
	load_from_archive(ar);
#endif
}

void Picviz::PVRoot::load_from_archive(PVCore::PVSerializeArchive_p ar)
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	ar->get_root()->object("root", *this, ARCHIVE_ROOT_DESC);
	_original_archive = ar;
#endif
}

PVCore::PVSerializeArchiveOptions_p Picviz::PVRoot::get_default_serialize_options()
{
	PVCore::PVSerializeArchiveOptions_p ar(new PVCore::PVSerializeArchiveOptions(PICVIZ_ARCHIVES_VERSION));
	ar->get_root()->object("root", *this, ARCHIVE_ROOT_DESC);
	return ar;
}
