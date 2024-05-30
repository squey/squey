//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/core/PVSerializeArchiveExceptions.h>

#include <squey/PVRoot.h>
#include <squey/PVScene.h>
#include <squey/PVView.h>
#include <squey/plugins.h>

/******************************************************************************
 *
 * Squey::PVRoot::PVRoot
 *
 *****************************************************************************/
Squey::PVRoot::PVRoot()
    : PVCore::PVDataTreeParent<PVScene, PVRoot>()
    , _current_scene(nullptr)
    , _current_source(nullptr)
    , _current_view(nullptr)
{
	reset_colors();
}

/******************************************************************************
 *
 * Squey::PVRoot::~PVRoot
 *
 *****************************************************************************/
Squey::PVRoot::~PVRoot()
{
	// Manually remove all child as we want to destroy them before ourself as
	// child may ask for modification in the Root (correlations, current_view, current_source, ...)
	remove_all_children();

	PVLOG_DEBUG("In PVRoot destructor\n");
}

void Squey::PVRoot::clear()
{
	remove_all_children();
	_current_scene = nullptr;
	_current_source = nullptr;
	_current_view = nullptr;
	_path.clear();
	_new_view_id = 0;
	reset_colors();
}

void Squey::PVRoot::reset_colors()
{
	_available_colors.clear();
	_available_colors << 0x9966CC << 0x6699CC << 0x778800 << 0xFFCC66 << 0x993366 << 0x999999
	                  << 0x339999 << 0xFF6633 << 0x99FFCC << 0xFFFF99;
	_used_colors.clear();
}

void Squey::PVRoot::select_view(PVView& view)
{
	assert(&view.get_parent<PVRoot>() == this);
	_current_view = &view;
	_current_scene = &view.get_parent<PVScene>();
	_current_source = &view.get_parent<PVSource>();

	_current_scene->set_last_active_source(_current_source);
	_current_source->set_last_active_view(&view);

	_scene_updated.emit();
}

void Squey::PVRoot::select_source(PVSource& src)
{
	assert(&src.get_parent<PVRoot>() == this);
	_current_source = &src;
	_current_view = src.last_active_view();
	_current_scene = &src.get_parent<PVScene>();

	_current_scene->set_last_active_source(&src);

	_scene_updated.emit();
}

void Squey::PVRoot::select_scene(PVScene& scene)
{
	assert(&scene.get_parent<PVRoot>() == this);
	_current_scene = &scene;
	_current_source = scene.last_active_source();
	if (_current_source) {
		_current_view = _current_source->last_active_view();
	}

	_scene_updated.emit();
}

void Squey::PVRoot::view_being_deleted(Squey::PVView* view)
{
	if (_current_view == view) {
		_current_view = nullptr;
	}
}

void Squey::PVRoot::scene_being_deleted(Squey::PVScene* scene)
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

void Squey::PVRoot::source_being_deleted(Squey::PVSource* src)
{
	if (_current_source == src) {
		_current_source = nullptr;
	}
	if (_current_view && &_current_view->get_parent<PVSource>() == src) {
		_current_view = nullptr;
	}
}

Squey::PVView* Squey::PVRoot::process_correlation(Squey::PVView* view)
{
	if (not _correlation_running) { // no indirect correlations to avoid potential infinite loops
		_correlation_running = true;
		Squey::PVView* view2 = correlations().process(view);
		_correlation_running = false;
		return view2;
	}

	return nullptr;
}

/******************************************************************************
 *
 * Squey::PVRoot::get_new_view_id
 *
 *****************************************************************************/
Squey::PVView::id_t Squey::PVRoot::get_new_view_id()
{
	return _new_view_id++;
}

/******************************************************************************
 *
 * Squey::PVRoot::get_new_view_color
 *
 *****************************************************************************/
QColor Squey::PVRoot::get_new_view_color()
{
	if (_available_colors.size() == 0) {
		std::swap(_available_colors, _used_colors);
	}
	QRgb color = _available_colors.at(0);
	_available_colors.pop_front();
	_used_colors << color;
	return color;
}

void Squey::PVRoot::save_to_file(PVCore::PVSerializeArchive& ar)
{
	auto root_obj = ar.get_root()->create_object("root");
	serialize_write(*root_obj);
}

void Squey::PVRoot::load_from_archive(PVCore::PVSerializeArchive& ar)
{
	auto root_ar = ar.get_root();
	if (ar.get_version() < 3) {
		throw PVCore::PVSerializeArchiveError("To make archives more robuste, we can't load data "
		                                      "from previous version of squey.");
	}
	auto root_obj = root_ar->create_object("root");
	serialize_read(*root_obj);
}

void Squey::PVRoot::serialize_write(PVCore::PVSerializeObject& so) const
{
	so.set_current_status("Saving root...");
	// Read the data colletions
	PVCore::PVSerializeObject_p list_obj = so.create_object("scene");
	int idx = 0;
	for (PVScene const* scene : get_children()) {
		PVCore::PVSerializeObject_p new_obj = list_obj->create_object(QString::number(idx++));
		scene->serialize_write(*new_obj);
	}
	so.attribute_write("scene_count", idx);
};

void Squey::PVRoot::serialize_read(PVCore::PVSerializeObject& so)
{
	so.set_current_status("Loading root...");
	// Read the data colletions
	PVCore::PVSerializeObject_p list_obj = so.create_object("scene");
	int scene_count = so.attribute_read<int>("scene_count");
	for (int idx = 0; idx < scene_count; idx++) {
		PVCore::PVSerializeObject_p new_obj = list_obj->create_object(QString::number(idx));
		PVScene::serialize_read(*new_obj, *this);
	}
}
