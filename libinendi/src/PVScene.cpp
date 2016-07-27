/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/hash_sharedptr.h>
#include <pvkernel/core/PVSerializeArchiveOptions.h>

#include <pvkernel/rush/PVNrawCacheManager.h>

#include <inendi/PVRoot.h>
#include <inendi/PVScene.h>
#include <inendi/PVSource.h>
#include <inendi/PVView.h>

#include <QFileInfo>

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
	QString name;
	so.attribute("name", name);
	PVScene& scene = root.emplace_add_child(name.toStdString());

	// Create a list of source
	PVCore::PVSerializeObject_p list_obj = so.create_object(
	    scene.get_children_serialize_name(), scene.get_children_description(), true, true);

	int idx = 0;
	try {
		while (true) {
			// FIXME It throws when there are no more data collections.
			// It should not be an exception as it is a normal behavior.
			PVCore::PVSerializeObject_p new_obj = list_obj->create_object(QString::number(idx++));
			PVSource::serialize_read(*new_obj, scene);
		}
	} catch (PVCore::PVSerializeArchiveErrorNoObject const&) {
	}

	return scene;
}

void Inendi::PVScene::serialize_write(PVCore::PVSerializeObject& so)
{
	QString name = QString::fromStdString(_name);
	so.attribute("name", name);

	// Read the data colletions
	PVCore::PVSerializeObject_p list_obj =
	    so.create_object(get_children_serialize_name(), get_children_description(), true, true);
	int idx = 0;
	for (PVSource* source : get_children()) {
		QString child_name = QString::number(idx++);
		PVCore::PVSerializeObject_p new_obj = list_obj->create_object(
		    child_name, QString::fromStdString(source->get_serialize_description()), false);
		source->serialize_write(*new_obj);
		new_obj->set_bound_obj(*source);
	}
}
