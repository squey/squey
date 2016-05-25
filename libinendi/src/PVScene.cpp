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
Inendi::PVScene::PVScene(Inendi::PVRoot* root, QString scene_name)
    : data_tree_scene_t(root), _last_active_src(nullptr), _name(scene_name)
{
}

/******************************************************************************
 *
 * Inendi::PVScene::~PVScene
 *
 *****************************************************************************/
Inendi::PVScene::~PVScene()
{
	remove_all_children();
	PVLOG_DEBUG("In PVScene destructor\n");
	PVRoot* root = get_parent<PVRoot>();
	if (root) {
		root->scene_being_deleted(this);
	}
}

Inendi::PVScene::list_sources_t Inendi::PVScene::get_sources(PVRush::PVInputType const& type) const
{
	children_t const& sources = get_children();
	list_sources_t ret;
	for (PVSource_sp const& src : sources) {
		if (*src->get_input_type() == type) {
			ret.push_back(src.get());
		}
	}
	return ret;
}

Inendi::PVSource* Inendi::PVScene::current_source()
{
	PVSource* cur_src = get_parent<PVRoot>()->current_source();
	if (cur_src->get_parent<PVScene>() == this) {
		return cur_src;
	}
	return nullptr;
}

Inendi::PVSource const* Inendi::PVScene::current_source() const
{
	PVSource const* cur_src = get_parent<PVRoot>()->current_source();
	if (cur_src->get_parent<PVScene>() == this) {
		return cur_src;
	}
	return nullptr;
}

Inendi::PVView* Inendi::PVScene::current_view()
{
	PVView* cur_view = get_parent<PVRoot>()->current_view();
	if (cur_view->get_parent<PVScene>() == this) {
		return cur_view;
	}
	return nullptr;
}

Inendi::PVView const* Inendi::PVScene::current_view() const
{
	PVView const* cur_view = get_parent<PVRoot>()->current_view();
	if (cur_view->get_parent<PVScene>() == this) {
		return cur_view;
	}
	return nullptr;
}

PVRush::PVInputType::list_inputs_desc
Inendi::PVScene::get_inputs_desc(PVRush::PVInputType const& type) const
{
	children_t const& sources = get_children();
	QSet<PVRush::PVInputDescription_p> ret_set;
	for (PVSource_sp const& src : sources) {
		if (*src->get_input_type() == type) {
			ret_set.unite(src->get_inputs().toSet());
		}
	}
	return ret_set.toList();
}

QList<PVRush::PVInputType_p> Inendi::PVScene::get_all_input_types() const
{
	QList<PVRush::PVInputType_p> ret;
	for (PVSource_sp const& src : get_children()) {
		PVRush::PVInputType_p in_type = src->get_input_type();
		bool found = false;
		for (PVRush::PVInputType_p const& known_in_t : ret) {
			if (known_in_t->registered_id() == in_type->registered_id()) {
				found = true;
				break;
			}
		}
		if (!found) {
			ret.push_back(in_type);
		}
	}
	return ret;
}

void Inendi::PVScene::serialize_read(PVCore::PVSerializeObject& so)
{
	// Get the list of input types
	QStringList input_types;
	so.list_attributes("types", input_types);

	// FIXME : Should check for 1 as there is a size attribute?
	if (input_types.size() == 0) {
		// No input types, thus no sources, thus nothing !
		return;
	}

	// Create a list of source
	PVCore::PVSerializeObject_p list_obj =
	    so.create_object(get_children_serialize_name(), get_children_description(), true, true);

	// Temporary list of input descriptions
	for (int i = 0; i < input_types.size(); i++) {
		// Create an object for the source
		PVCore::PVSerializeObject_p new_obj = list_obj->create_object(QString::number(i));

		QString src_name;
		new_obj->attribute("source-plugin", src_name);
		// FIXME : Handle error when source name if not correct
		PVRush::PVSourceCreator_p sc_lib =
		    LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name(src_name);

		// Get the input type lib object
		QString const& type_name = input_types.at(i);
		// FIXME : We should check for type_name validity if archive was manually changed.
		PVRush::PVInputType_p int_lib =
		    LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name(type_name);

		// Get the inputs list object for that input type
		PVRush::PVInputType::list_inputs_desc inputs_for_type;

		// Get back the inputs
		int_lib->serialize_inputs(so, type_name, inputs_for_type);

		PVRush::PVFormat format;
		new_obj->object("format", format);

		// Get the state of the extractor
		chunk_index start, nlines;
		new_obj->attribute("index_start", start);
		new_obj->attribute("nlines", nlines);

		PVCore::PVSharedPtr<PVSource> source =
		    emplace_add_child(inputs_for_type, sc_lib, format, start, nlines);

		QString nraw_folder;
		new_obj->attribute("nraw_path", nraw_folder, QString());

		if (not nraw_folder.isEmpty()) {
			QString user_based_nraw_dir = PVRush::PVNrawCacheManager::nraw_dir() +
			                              QDir::separator() + QDir(nraw_folder).dirName();
			QFileInfo fi(user_based_nraw_dir);
			if (fi.exists() == true && fi.isDir() == true) {
				nraw_folder = user_based_nraw_dir;
			} else {
				nraw_folder = QString();
			}
		}
		source->set_nraw_folder(nraw_folder);
		source->serialize(*new_obj, so.get_version());
		new_obj->_bound_obj = source.get();
		new_obj->_bound_obj_type = typeid(PVSource);
	}
}

void Inendi::PVScene::serialize_write(PVCore::PVSerializeObject& so)
{
	// First serialize the input descriptions.
	QList<PVRush::PVInputType_p> in_types(get_all_input_types());
	QStringList in_types_str;
	for (PVRush::PVInputType_p const& in_t : in_types) {
		list_sources_t sources = get_sources(*in_t);

		// Safety check
		if (sources.size() == 0) {
			continue;
		}

		// FIXME: Files are not saved in source while they certainly should.
		PVRush::PVInputType::list_inputs_desc inputs = get_inputs_desc(*in_t);
		in_t->serialize_inputs(so, in_t->registered_name(), inputs);
		in_types_str.push_back(in_t->registered_name());
	}

	so.list_attributes("types", in_types_str);
	so.attribute("name", _name);

	// Read the data colletions
	PVCore::PVSerializeObject_p list_obj =
	    so.create_object(get_children_serialize_name(), get_children_description(), true, true);
	int idx = 0;
	for (PVCore::PVSharedPtr<PVSource> source : get_children()) {
		QString child_name = QString::number(idx++);
		PVCore::PVSerializeObject_p new_obj =
		    list_obj->create_object(child_name, source->get_serialize_description(), false);
		source->serialize(*new_obj, so.get_version());
		new_obj->_bound_obj = source.get();
		new_obj->_bound_obj_type = typeid(PVSource);
	}
}
