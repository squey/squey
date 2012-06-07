//! \file PVScene.h
//! $Id: PVScene.h 2875 2011-05-19 04:18:05Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVSCENE_H
#define PICVIZ_PVSCENE_H

#include <QString>

#include <pvkernel/core/PVDataTreeObject.h>
#include <pvkernel/core/general.h>
#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/core/PVSerializeArchiveOptions_types.h>
#include <pvkernel/rush/PVInputDescription.h>
#include <pvkernel/rush/PVInputType.h>
#include <picviz/PVAD2GView.h>
#include <picviz/PVPtrObjects.h>
#include <picviz/PVSource_types.h>
#include <picviz/PVView_types.h>


#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>

#define PICVIZ_SCENE_ARCHIVE_EXT "pv"
#define PICVIZ_SCENE_ARCHIVE_FILTER "Picviz project files (*." PICVIZ_SCENE_ARCHIVE_EXT ")"

namespace Picviz {

class PVSource;

/**
 * \class PVScene
 */
typedef typename PVCore::PVDataTreeObject<PVRoot, PVSource> data_tree_scene_t;
class LibPicvizDecl PVScene: public data_tree_scene_t, public boost::enable_shared_from_this<PVScene>
{
	friend class PVCore::PVSerializeObject;
	friend class PVSource;
	friend class PVView;
public:
	typedef boost::shared_ptr<PVScene> p_type;
	typedef QList<PVSource_p> list_sources_t;
	typedef QList<PVView_p> list_views_t;
private:
	// PVRush::list_inputs is QList<PVRush::PVInputDescription_p>
	typedef std::map<PVRush::PVInputType::base_registrable, std::pair<list_sources_t, PVRush::PVInputType::list_inputs> > hash_type_sources_t;
	typedef std::map<PVRush::PVInputType::base_registrable, PVCore::PVSerializeObject_p> hash_type_so_inputs;
public:
	
	PVScene(QString scene_name, PVRoot* parent);
	~PVScene();

public:
	PVRoot* get_root();

public:
	PVCore::PVSerializeArchiveOptions_p get_default_serialize_options();
	void save_to_file(QString const& path, PVCore::PVSerializeArchiveOptions_p options = PVCore::PVSerializeArchiveOptions_p(), bool save_everything = false);
	void load_from_file(QString const& path);
	void load_from_archive(PVCore::PVSerializeArchive_p ar);

public:
	void add_source(PVSource_p src);
	bool del_source(const PVSource* src);
	
	list_sources_t get_sources(PVRush::PVInputType const& type) const;
	list_sources_t get_all_sources() const;
	list_views_t get_all_views() const;

	inline PVAD2GView& get_ad2g_view() { return *_ad2g_view; }
	inline PVAD2GView const& get_ad2g_view() const { return *_ad2g_view; }
	inline PVAD2GView_p get_ad2g_view_p() { return _ad2g_view; }

	bool is_empty() { return _sources.size() == 0; }

protected:
	int32_t get_new_view_id() const;
	void set_views_id();

protected:
	// From PVView
	void user_modified_sel(Picviz::PVView* org, QList<Picviz::PVView*>* changed_views = NULL);

	// Serialization
	void serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);
	void serialize_write(PVCore::PVSerializeObject& so);
	PVSERIALIZEOBJECT_SPLIT

	PVCore::PVSerializeObject_p get_so_inputs(PVSource const& src);

private:
	hash_type_sources_t _sources;
	hash_type_so_inputs _so_inputs;

	PVRoot* _root;
	QString _name;

	// This is a shared pointer for current issues with the widget (which will be deleted by Qt *after*
	// this object).
	PVAD2GView_p _ad2g_view;

	PVCore::PVSerializeArchive_p _original_archive;
};

typedef PVScene::p_type PVScene_p;

}

#endif	/* PICVIZ_PVSCENE_H */
