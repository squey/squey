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

#include <inendi/PVRoot.h>   // for PVRoot
#include <inendi/PVScene.h>  // for PVScene
#include <inendi/PVSource.h> // for PVSource
#include <inendi/PVView.h>   // for PVView

#include <pvkernel/core/PVDataTreeObject.h>  // for PVDataTreeChild
#include <pvkernel/core/PVLogger.h>          // for PVLOG_DEBUG
#include <pvkernel/core/PVSerializeObject.h> // for PVSerializeObject, etc

#include <QString> // for QString

#include <memory> // for __shared_ptr
#include <string> // for string

#define ARCHIVE_SCENE_DESC (QObject::tr("Workspace"))
/******************************************************************************
 *
 * Inendi::PVScene::PVScene
 *
 *****************************************************************************/
Inendi::PVScene::PVScene(Inendi::PVRoot& root, std::string const& scene_name)
    : PVCore::PVDataTreeChild<PVRoot, PVScene>(root), _last_active_src(nullptr), _name(scene_name)
{
}

/******************************************************************************
 *
 * Inendi::PVScene::~PVScene
 *
 *****************************************************************************/
Inendi::PVScene::~PVScene()
{
	PVLOG_DEBUG("In PVScene destructor\n");
	get_parent<PVRoot>().scene_being_deleted(this);
}

Inendi::PVSource* Inendi::PVScene::current_source()
{
	PVSource* cur_src = get_parent<PVRoot>().current_source();
	if (&cur_src->get_parent<PVScene>() == this) {
		return cur_src;
	}
	return nullptr;
}

Inendi::PVSource const* Inendi::PVScene::current_source() const
{
	PVSource const* cur_src = get_parent<PVRoot>().current_source();
	if (&cur_src->get_parent<PVScene>() == this) {
		return cur_src;
	}
	return nullptr;
}

Inendi::PVView* Inendi::PVScene::current_view()
{
	PVView* cur_view = get_parent<PVRoot>().current_view();
	if (&cur_view->get_parent<PVScene>() == this) {
		return cur_view;
	}
	return nullptr;
}

Inendi::PVView const* Inendi::PVScene::current_view() const
{
	PVView const* cur_view = get_parent<PVRoot>().current_view();
	if (&cur_view->get_parent<PVScene>() == this) {
		return cur_view;
	}
	return nullptr;
}

Inendi::PVScene& Inendi::PVScene::serialize_read(PVCore::PVSerializeObject& so,
                                                 Inendi::PVRoot& root)
{
	so.set_current_status("Loading scene...");
	auto name = so.attribute_read<QString>("name");
	PVScene& scene = root.emplace_add_child(name.toStdString());

	// Create a list of source
	PVCore::PVSerializeObject_p list_obj = so.create_object("source");

	int source_count = so.attribute_read<int>("source_count");
	for (int idx = 0; idx < source_count; idx++) {
		PVCore::PVSerializeObject_p new_obj = list_obj->create_object(QString::number(idx));
		PVSource::serialize_read(*new_obj, scene);
	}

	return scene;
}

void Inendi::PVScene::serialize_write(PVCore::PVSerializeObject& so) const
{
	so.set_current_status("Saving scene...");
	QString name = QString::fromStdString(_name);
	so.attribute_write("name", name);

	// Read the data colletions
	PVCore::PVSerializeObject_p list_obj = so.create_object("source");
	int idx = 0;
	for (PVSource const* source : get_children()) {
		PVCore::PVSerializeObject_p new_obj = list_obj->create_object(QString::number(idx++));
		source->serialize_write(*new_obj);
	}
	so.attribute_write("source_count", idx);
}
