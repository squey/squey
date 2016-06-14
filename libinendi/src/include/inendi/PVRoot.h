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

#include <sigc++/sigc++.h>

#include <pvkernel/core/PVDataTreeObject.h>
#include <pvkernel/core/PVSerializeObject.h>
#include <pvkernel/core/PVSerializeArchiveOptions.h>

#include <inendi/PVScene.h>
#include <inendi/PVCorrelationEngine.h>

#define INENDI_ROOT_ARCHIVE_EXT "pvi"
#define INENDI_ROOT_ARCHIVE_FILTER "INENDI investigation files (*." INENDI_ROOT_ARCHIVE_EXT ")"

namespace Inendi
{

class PVView;

/**
 * \class PVRoot
 */
class PVRoot : public PVCore::PVDataTreeParent<PVScene, PVRoot>,
               public PVCore::PVEnableSharedFromThis<PVRoot>
{
  public:
	friend class PVView;
	friend class PVScene;
	friend class PVSource;
	friend class PVCore::PVSerializeObject;

  public:
	PVRoot();
	~PVRoot();

  public:
	bool is_empty() const { return size() == 0; }
	void clear();
	void reset_colors();

  public:
	int32_t get_new_view_id();
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

  public:
	void save_to_file(
	    QString const& path,
	    PVCore::PVSerializeArchiveOptions_p options = PVCore::PVSerializeArchiveOptions_p(),
	    bool save_everything = false);
	void load_from_archive(PVCore::PVSerializeArchive_p ar);
	PVCore::PVSerializeArchiveOptions_p get_default_serialize_options();

	void set_path(QString path) { _path = path; }
	const QString& get_path() const { return _path; }

  public:
	std::string get_serialize_description() const override { return "Investigation"; }

	virtual QString get_children_description() const { return "Data collection(s)"; }
	virtual QString get_children_serialize_name() const { return "data-collections"; }

  public:
	PVCorrelationEngine& correlations() { return _correlations; }
	const PVCorrelationEngine& correlations() const { return _correlations; }
	Inendi::PVView* process_correlation(Inendi::PVView* view);

  protected:
	void view_being_deleted(Inendi::PVView* view);
	void scene_being_deleted(Inendi::PVScene* view);
	void source_being_deleted(Inendi::PVSource* view);

  protected:
	// Serialization
	void serialize_write(PVCore::PVSerializeObject& so);

	/**
	 * Read Childs from pvi
	 *
	 * root
	 *   |->data-colletions
	 *       |-> 0
	 *       |-> 1
	 *       |-> ...
	 */
	void serialize_read(PVCore::PVSerializeObject& so);
	PVSERIALIZEOBJECT_SPLIT

  public:
	sigc::signal<void> _scene_updated;

  private:
	PVScene* _current_scene;
	PVSource* _current_source;
	PVView* _current_view;

	PVCorrelationEngine _correlations;

	QList<QRgb> _available_colors;
	QList<QRgb> _used_colors;

  private:
	int _new_view_id = 0;

	QString _path;
	bool _correlation_running = false;
};
}

#endif /* INENDI_PVROOT_H */
