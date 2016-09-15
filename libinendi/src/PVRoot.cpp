/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVSerializeArchiveExceptions.h>

#include <inendi/PVRoot.h>
#include <inendi/PVScene.h>
#include <inendi/PVView.h>
#include <inendi/plugins.h>

/******************************************************************************
 *
 * Inendi::PVRoot::PVRoot
 *
 *****************************************************************************/
Inendi::PVRoot::PVRoot()
    : PVCore::PVDataTreeParent<PVScene, PVRoot>()
    , _current_scene(nullptr)
    , _current_source(nullptr)
    , _current_view(nullptr)
{
	reset_colors();
}

/******************************************************************************
 *
 * Inendi::PVRoot::~PVRoot
 *
 *****************************************************************************/
Inendi::PVRoot::~PVRoot()
{
	// Manually remove all child as we want to destroy them before ourself as
	// child may ask for modification in the Root (correlations, current_view, current_source, ...)
	remove_all_children();

	PVLOG_DEBUG("In PVRoot destructor\n");
}

void Inendi::PVRoot::clear()
{
	remove_all_children();
	_current_scene = nullptr;
	_current_source = nullptr;
	_current_view = nullptr;
	_path.clear();
	_new_view_id = 0;
	reset_colors();
}

void Inendi::PVRoot::reset_colors()
{
	_available_colors.clear();
	_available_colors << 0x9966CC << 0x6699CC << 0x778800 << 0xFFCC66 << 0x993366 << 0x999999
	                  << 0x339999 << 0xFF6633 << 0x99FFCC << 0xFFFF99;
	_used_colors.clear();
}

void Inendi::PVRoot::select_view(PVView& view)
{
	assert(&view.get_parent<PVRoot>() == this);
	_current_view = &view;
	_current_scene = &view.get_parent<PVScene>();
	_current_source = &view.get_parent<PVSource>();

	_current_scene->set_last_active_source(_current_source);
	_current_source->set_last_active_view(&view);

	_scene_updated.emit();
}

void Inendi::PVRoot::select_source(PVSource& src)
{
	assert(&src.get_parent<PVRoot>() == this);
	_current_source = &src;
	_current_view = src.last_active_view();
	_current_scene = &src.get_parent<PVScene>();

	_current_scene->set_last_active_source(&src);

	_scene_updated.emit();
}

void Inendi::PVRoot::select_scene(PVScene& scene)
{
	assert(&scene.get_parent<PVRoot>() == this);
	_current_scene = &scene;
	_current_source = scene.last_active_source();
	if (_current_source) {
		_current_view = _current_source->last_active_view();
	}

	_scene_updated.emit();
}

void Inendi::PVRoot::view_being_deleted(Inendi::PVView* view)
{
	if (_current_view == view) {
		_current_view = nullptr;
	}
}

void Inendi::PVRoot::scene_being_deleted(Inendi::PVScene* scene)
{
	if (_current_scene == scene) {
		_current_scene = nullptr;
	}
	if (_current_source && &_current_source->get_parent<PVScene>() == scene) {
		_current_source = nullptr;
	}
	if (_current_view && &_current_view->get_parent<PVScene>() == scene) {
		_current_view = nullptr;
	}
}

void Inendi::PVRoot::source_being_deleted(Inendi::PVSource* src)
{
	if (_current_source == src) {
		_current_source = nullptr;
	}
	if (_current_view && &_current_view->get_parent<PVSource>() == src) {
		_current_view = nullptr;
	}
}

Inendi::PVView* Inendi::PVRoot::process_correlation(Inendi::PVView* view)
{
	if (not _correlation_running) { // no indirect correlations to avoid potential infinite loops
		_correlation_running = true;
		Inendi::PVView* view2 = correlations().process(view);
		_correlation_running = false;
		return view2;
	}

	return nullptr;
}

/******************************************************************************
 *
 * Inendi::PVRoot::get_new_view_id
 *
 *****************************************************************************/
Inendi::PVView::id_t Inendi::PVRoot::get_new_view_id()
{
	return _new_view_id++;
}

/******************************************************************************
 *
 * Inendi::PVRoot::get_new_view_color
 *
 *****************************************************************************/
QColor Inendi::PVRoot::get_new_view_color()
{
	if (_available_colors.size() == 0) {
		std::swap(_available_colors, _used_colors);
	}
	QRgb color = _available_colors.at(0);
	_available_colors.pop_front();
	_used_colors << color;
	return color;
}

void Inendi::PVRoot::save_to_file(PVCore::PVSerializeArchive& ar)
{
	auto root_obj = ar.get_root()->create_object("root");
	serialize_write(*root_obj);
}

void Inendi::PVRoot::load_from_archive(PVCore::PVSerializeArchive& ar)
{
	auto root_ar = ar.get_root();
	if (ar.get_version() < 3) {
		throw PVCore::PVSerializeArchiveError("To make archives more robuste, we can't load data "
		                                      "from previous version of inspector.");
	}
	auto root_obj = root_ar->create_object("root");
	serialize_read(*root_obj);
}

void Inendi::PVRoot::serialize_write(PVCore::PVSerializeObject& so)
{
	so.set_current_status("Serialize Root.");
	// Read the data colletions
	PVCore::PVSerializeObject_p list_obj = so.create_object("scene");
	int idx = 0;
	for (PVScene* scene : get_children()) {
		PVCore::PVSerializeObject_p new_obj = list_obj->create_object(QString::number(idx++));
		scene->serialize_write(*new_obj);
	}
	so.attribute("scene_count", idx);
};

void Inendi::PVRoot::serialize_read(PVCore::PVSerializeObject& so)
{
	so.set_current_status("Loading root");
	// Read the data colletions
	PVCore::PVSerializeObject_p list_obj = so.create_object("scene");
	int scene_count;
	so.attribute("scene_count", scene_count);
	for (int idx = 0; idx < scene_count; idx++) {
		PVCore::PVSerializeObject_p new_obj = list_obj->create_object(QString::number(idx));
		PVScene::serialize_read(*new_obj, *this);
	}
}
