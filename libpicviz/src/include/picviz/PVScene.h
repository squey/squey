/**
 * \file PVScene.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PICVIZ_PVSCENE_H
#define PICVIZ_PVSCENE_H

#include <QString>

#include <pvkernel/core/PVDataTreeObject.h>
#include <pvkernel/core/general.h>
#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/core/PVSerializeArchiveOptions_types.h>
#include <pvkernel/rush/PVInputDescription.h>
#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/rush/PVSourceDescription.h>
#include <picviz/PVAD2GView.h>
#include <picviz/PVPtrObjects.h>
#include <picviz/PVSource_types.h>
#include <picviz/PVRoot.h>
#include <picviz/PVView_types.h>
#include <picviz/PVScene_types.h>


#include <boost/shared_ptr.hpp>

#define PICVIZ_SCENE_ARCHIVE_EXT "pv"
#define PICVIZ_SCENE_ARCHIVE_FILTER "Picviz project files (*." PICVIZ_SCENE_ARCHIVE_EXT ")"

namespace Picviz {

class PVSource;

/**
 * \class PVScene
 */
typedef typename PVCore::PVDataTreeObject<PVRoot, PVSource> data_tree_scene_t;
class LibPicvizDecl PVScene: public data_tree_scene_t
{
	friend class PVCore::PVSerializeObject;
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
	void select_source(PVSource* source);

	PVView* current_view();
	PVView const* current_view() const;
	void select_view(PVView& view);

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
	Picviz::PVSource_p add_source_from_description(const PVRush::PVSourceDescription& descr);

	virtual QString get_serialize_description() const { return "Scene"; }

protected:
	int32_t get_new_view_id() const;
	void set_views_id();

	QColor get_new_view_color() const;

	virtual QString get_children_description() const { return "Source(s)"; }
	virtual QString get_children_serialize_name() const { return "sources"; }

	QList<PVRush::PVInputType_p> get_all_input_types() const;

protected:
	// Events
	void child_about_to_be_removed(PVSource& src);
	void child_added(PVSource& src);

protected:
	// From PVView
	void user_modified_sel(Picviz::PVView* org, QList<Picviz::PVView*>* changed_views = NULL);

	// Serialization
	void serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);
	void serialize_write(PVCore::PVSerializeObject& so);
	PVSERIALIZEOBJECT_SPLIT

	PVCore::PVSerializeObject_p get_so_inputs(PVSource const& src);

private:
	PVView* _current_view = nullptr;
	PVSource* _current_source = nullptr;
	hash_type_so_inputs _so_inputs;

	QString _path;
	QString _name;

	PVCore::PVSerializeArchive_p _original_archive;

	QRgb _view_colors[4] = { 0x9966CC, 0x6699CC, 0x778800, 0xFFCC66 } ;

};

typedef PVScene::p_type  PVScene_p;
typedef PVScene::wp_type PVScene_wp;

}

#endif	/* PICVIZ_PVSCENE_H */
