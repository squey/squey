/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVSCENE_H
#define INENDI_PVSCENE_H

#include <QString>

#include <pvkernel/core/PVDataTreeObject.h>
#include <pvkernel/core/general.h>
#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/core/PVSerializeArchiveOptions_types.h>
#include <pvkernel/rush/PVInputDescription.h>
#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/rush/PVSourceDescription.h>
#include <inendi/PVPtrObjects.h>
#include <inendi/PVSource_types.h>
#include <inendi/PVRoot.h>
#include <inendi/PVView_types.h>
#include <inendi/PVScene_types.h>

#define INENDI_SCENE_ARCHIVE_EXT "pv"
#define INENDI_SCENE_ARCHIVE_FILTER "INENDI project files (*." INENDI_SCENE_ARCHIVE_EXT ")"

namespace Inendi {

class PVSource;

/**
 * \class PVScene
 */
typedef typename PVCore::PVDataTreeObject<PVRoot, PVSource> data_tree_scene_t;
class PVScene: public data_tree_scene_t
{
	friend class PVCore::PVSerializeObject;
	friend class PVRoot;
	friend class PVSource;
	friend class PVView;
	friend class PVCore::PVDataTreeAutoShared<PVScene>;
public:
	typedef QList<PVSource*> list_sources_t;
private:
	// PVRush::list_inputs is QList<PVRush::PVInputDescription_p>
	typedef std::map<PVRush::PVInputType::base_registrable, PVCore::PVSerializeObject_p> hash_type_so_inputs;

protected:
	PVScene(QString scene_path = QString());

public:
	~PVScene();

public:
	void set_name(QString name) { _name = name; }
	const QString & get_name() const { return _name; }
	void set_path(QString path) { _path = path; }
	const QString & get_path() const { return _path; }

	PVSource* current_source();
	PVSource const* current_source() const;

	PVView* current_view();
	PVView const* current_view() const;

	inline PVSource* last_active_source() { return _last_active_src; }
	inline PVSource const* last_active_source() const { return _last_active_src; }

public:
	PVCore::PVSerializeArchiveOptions_p get_default_serialize_options();
	void save_to_file(QString const& path, PVCore::PVSerializeArchiveOptions_p options = PVCore::PVSerializeArchiveOptions_p(), bool save_everything = false);
	void load_from_file(QString const& path);
	void load_from_archive(PVCore::PVSerializeArchive_p ar);

public:
	list_sources_t get_sources(PVRush::PVInputType const& type) const;
	PVRush::PVInputType::list_inputs_desc get_inputs_desc(PVRush::PVInputType const& type) const;

	inline bool is_empty() const { return get_children().size() == 0; }

	void add_source(PVSource_p const& src);
	Inendi::PVSource_p add_source_from_description(const PVRush::PVSourceDescription& descr);

	virtual QString get_serialize_description() const { return get_name(); }

protected:
	/*int32_t get_new_view_id() const;
	void set_views_id();

	QColor get_new_view_color() const;*/

	virtual QString get_children_description() const { return "Source(s)"; }
	virtual QString get_children_serialize_name() const { return "sources"; }

	QList<PVRush::PVInputType_p> get_all_input_types() const;

	inline void set_last_active_source(PVSource* src) { _last_active_src = src; }

protected:
	// Events
	void child_about_to_be_removed(PVSource& src);
	void child_added(PVSource& src);

protected:
	// Serialization
	void serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);
	void serialize_write(PVCore::PVSerializeObject& so);
	PVSERIALIZEOBJECT_SPLIT

	PVCore::PVSerializeObject_p get_so_inputs(PVSource const& src);

private:
	Inendi::PVSource* _last_active_src;

	hash_type_so_inputs _so_inputs;
	PVCore::PVSerializeArchive_p _original_archive;

	QString _path;
	QString _name;
};

typedef PVScene::p_type  PVScene_p;
typedef PVScene::wp_type PVScene_wp;

}

#endif	/* INENDI_PVSCENE_H */