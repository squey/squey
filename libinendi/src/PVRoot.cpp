/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVSerializeArchiveZip.h>

#include <inendi/PVRoot.h>
#include <inendi/PVScene.h>
#include <inendi/PVView.h>
#include <inendi/plugins.h>

#define ARCHIVE_ROOT_DESC (QObject::tr("Solution"))

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
 * Inendi::PVRoot::PVRoot
 *
 *****************************************************************************/
Inendi::PVRoot::~PVRoot()
{
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
	assert(view.get_parent<PVRoot>() == this);
	_current_view = &view;
	_current_scene = view.get_parent<PVScene>();
	_current_source = view.get_parent<PVSource>();

	_current_scene->set_last_active_source(_current_source);
	_current_source->set_last_active_view(&view);
}

void Inendi::PVRoot::select_source(PVSource& src)
{
	assert(src.get_parent<PVRoot>() == this);
	_current_source = &src;
	_current_view = src.last_active_view();
	_current_scene = src.get_parent<PVScene>();

	_current_scene->set_last_active_source(&src);
}

void Inendi::PVRoot::select_scene(PVScene& scene)
{
	assert(scene.get_parent<PVRoot>() == this);
	_current_scene = &scene;
	_current_source = scene.last_active_source();
	if (_current_source) {
		_current_view = _current_source->last_active_view();
	}
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
	if (_current_source && _current_source->get_parent<PVScene>() == scene) {
		_current_source = nullptr;
	}
	if (_current_view && _current_view->get_parent<PVScene>() == scene) {
		_current_view = nullptr;
	}
}

void Inendi::PVRoot::source_being_deleted(Inendi::PVSource* src)
{
	if (_current_source == src) {
		_current_source = nullptr;
	}
	if (_current_view && _current_view->get_parent<PVSource>() == src) {
		_current_view = nullptr;
	}
}

Inendi::PVView* Inendi::PVRoot::process_correlation(Inendi::PVView* view)
{
	if (not _correlation_running) { // no indirect correlations to avoid potential infinite loops
		_correlation_running = true;
		Inendi::PVView* view2 = correlations().process(view);

		if (view2) {
			view2->process_from_selection();
		}

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

void Inendi::PVRoot::save_to_file(QString const& path,
                                  PVCore::PVSerializeArchiveOptions_p options,
                                  bool save_everything)
{
	set_path(path);
	PVCore::PVSerializeArchive_p ar(new PVCore::PVSerializeArchiveZip(
	    path, PVCore::PVSerializeArchive::write, INENDI_ARCHIVES_VERSION));
	if (options) {
		ar->set_options(options);
	}
	ar->set_save_everything(save_everything);
	ar->get_root()->object("root", *this, ARCHIVE_ROOT_DESC);
	ar->finish();
}

void Inendi::PVRoot::load_from_archive(PVCore::PVSerializeArchive_p ar)
{
	ar->get_root()->object("root", *this, ARCHIVE_ROOT_DESC);
}

PVCore::PVSerializeArchiveOptions_p Inendi::PVRoot::get_default_serialize_options()
{
	PVCore::PVSerializeArchiveOptions_p ar(
	    new PVCore::PVSerializeArchiveOptions(INENDI_ARCHIVES_VERSION));
	ar->get_root()->object("root", *this, ARCHIVE_ROOT_DESC);
	return ar;
}

void Inendi::PVRoot::serialize_write(PVCore::PVSerializeObject& so)
{
	// Read the data colletions
	PVCore::PVSerializeObject_p list_obj =
	    so.create_object(get_children_serialize_name(), get_children_description(), true, true);
	int idx = 0;
	for (PVScene_p scene : get_children()) {
		QString child_name = QString::number(idx++);
		PVCore::PVSerializeObject_p new_obj =
		    list_obj->create_object(child_name, scene->get_serialize_description(), false);
		scene->serialize(*new_obj, so.get_version());
		new_obj->_bound_obj = scene.get();
		new_obj->_bound_obj_type = typeid(PVScene);
	}
};

void Inendi::PVRoot::serialize_read(PVCore::PVSerializeObject& so)
{
	// Read the data colletions
	PVCore::PVSerializeObject_p list_obj =
	    so.create_object(get_children_serialize_name(), get_children_description(), true, true);
	int idx = 0;
	try {
		while (true) {
			// FIXME It throws when there are no more data collections.
			// It should not be an exception as it is a normal behavior.
			PVCore::PVSerializeObject_p new_obj = list_obj->create_object(QString::number(idx));
			QString name;
			new_obj->attribute("name", name);
			PVScene_p scene = emplace_add_child(name);
			scene->serialize(*new_obj, so.get_version());
			new_obj->_bound_obj = scene.get();
			new_obj->_bound_obj_type = typeid(PVScene);
			idx++;
		}
	} catch (PVCore::PVSerializeArchiveErrorNoObject const&) {
		return;
	}
}
