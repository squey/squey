/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVROOT_H
#define INENDI_PVROOT_H

#include <QColor>
#include <QList>
#include <QStringList>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVDataTreeObject.h>
#include <pvkernel/core/PVSerializeObject.h>
#include <pvkernel/core/PVSerializeArchiveOptions.h>

#include <inendi/PVRoot_types.h>
#include <inendi/PVPtrObjects.h> // For PVScene_p

#define INENDI_ROOT_ARCHIVE_EXT "pvi"
#define INENDI_ROOT_ARCHIVE_FILTER "INENDI investigation files (*." INENDI_ROOT_ARCHIVE_EXT ")"

// Plugins prefix
#define LAYER_FILTER_PREFIX "layer_filter"
#define MAPPING_FILTER_PREFIX "mapping_filter"
#define PLOTTING_FILTER_PREFIX "plotting_filter"
#define AXIS_COMPUTATION_PLUGINS_PREFIX "axis_computation"
#define SORTING_FUNCTIONS_PLUGINS_PREFIX "sorting"

namespace Inendi
{

class PVView;

/**
 * \class PVRoot
 */
typedef typename PVCore::PVDataTreeObject<PVCore::PVDataTreeNoParent<PVRoot>, PVScene>
    data_tree_root_t;
class PVRoot : public data_tree_root_t
{
  public:
	friend class PVView;
	friend class PVScene;
	friend class PVSource;
	friend class PVCore::PVSerializeObject;
	// typedef std::shared_ptr<PVRoot> p_type;

  public:
	PVRoot();
	~PVRoot();

  public:
	bool is_empty() const { return get_children_count() == 0; }
	void clear();
	void reset_colors();

  public:
	int32_t get_new_view_id();
	void set_views_id();
	QColor get_new_view_color();

  public:
	void select_view(PVView& view);
	void select_scene(PVScene& scene);
	void select_source(PVSource& source);

	Inendi::PVView* current_view() { return _current_view; }
	Inendi::PVView const* current_view() const { return _current_view; }

	Inendi::PVScene* current_scene() { return _current_scene; }
	Inendi::PVScene const* current_scene() const { return _current_scene; }

	Inendi::PVSource* current_source() { return _current_source; }
	Inendi::PVSource const* current_source() const { return _current_source; }

	PVScene** get_current_scene_hive_property() { return &_current_scene; }
	PVView** get_current_view_hive_property() { return &_current_view; }
	PVSource** get_current_source_hive_property() { return &_current_source; }

	PVScene* get_scene_from_path(const QString& path);

  public:
	void save_to_file(
	    QString const& path,
	    PVCore::PVSerializeArchiveOptions_p options = PVCore::PVSerializeArchiveOptions_p(),
	    bool save_everything = false);
	void load_from_file(QString const& path);
	void load_from_archive(PVCore::PVSerializeArchive_p ar);
	PVCore::PVSerializeArchiveOptions_p get_default_serialize_options();

	void set_path(QString path) { _path = path; }
	const QString& get_path() const { return _path; }

  public:
	virtual QString get_serialize_description() const { return "Investigation"; }

	virtual QString get_children_description() const { return "Data collection(s)"; }
	virtual QString get_children_serialize_name() const { return "data-collections"; }

  protected:
	void view_being_deleted(Inendi::PVView* view);
	void scene_being_deleted(Inendi::PVScene* view);
	void source_being_deleted(Inendi::PVSource* view);

  protected:
	// Serialization
	void serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);
	void serialize_write(PVCore::PVSerializeObject& so);
	PVSERIALIZEOBJECT_SPLIT

  private:
	PVScene* _current_scene;
	PVSource* _current_source;
	PVView* _current_view;

	QList<QRgb> _available_colors;
	QList<QRgb> _used_colors;

  private:
	int _new_view_id = 0;

	PVCore::PVSerializeArchive_p _original_archive;
	QString _path;
};

typedef PVRoot::p_type PVRoot_p;
}

#endif /* INENDI_PVROOT_H */
