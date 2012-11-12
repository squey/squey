/**
 * \file PVRoot.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PICVIZ_PVROOT_H
#define PICVIZ_PVROOT_H

#include <QList>
#include <QRgb>
#include <QStringList>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVDataTreeObject.h>
#include <pvkernel/core/PVSerializeObject.h>
#include <pvkernel/core/PVSerializeArchiveOptions.h>

#include <picviz/PVAD2GView.h>
#include <picviz/PVRoot_types.h>
#include <picviz/PVPtrObjects.h> // For PVScene_p

#include <boost/shared_ptr.hpp>

#define PICVIZ_ROOT_ARCHIVE_EXT "pvl"
#define PICVIZ_ROOT_ARCHIVE_FILTER "Picviz solution files (*." PICVIZ_SCENE_ARCHIVE_EXT ")"

// Plugins prefix
#define LAYER_FILTER_PREFIX "layer_filter"
#define MAPPING_FILTER_PREFIX "mapping_filter"
#define PLOTTING_FILTER_PREFIX "plotting_filter"
#define ROW_FILTER_PREFIX "row_filter"
#define AXIS_COMPUTATION_PLUGINS_PREFIX "axis_computation"
#define SORTING_FUNCTIONS_PLUGINS_PREFIX "sorting"

namespace Picviz {

class PVView;

/**
 * \class PVRoot
 */
typedef typename PVCore::PVDataTreeObject<PVCore::PVDataTreeNoParent<PVRoot>, PVScene> data_tree_root_t;
class LibPicvizDecl PVRoot : public data_tree_root_t {
public:
	friend class PVView;
	friend class PVScene;
	friend class PVSource;
	friend class PVCore::PVSerializeObject;
	//typedef boost::shared_ptr<PVRoot> p_type;
	typedef std::list<PVAD2GView_p> correlations_t;

public:
	PVRoot();
	~PVRoot();

public:
	PVView* current_view();
	PVView const* current_view() const;
	void select_view(PVView& view);
	bool is_empty() const { return get_children_count() == 0; }

public:
	int32_t get_new_view_id() const;
	void set_views_id();
	QColor get_new_view_color() const;

public:
	PVAD2GView_p get_correlation(int index);
	void select_correlation(int index)
	{
		if (index <= -1) {
			_current_correlation = nullptr;
		}
		else {
			_current_correlation = get_correlation(index).get();
		}
	}
	PVAD2GView* add_correlation(const QString & name);
	void add_correlations(correlations_t const& corrs);
	void delete_correlation(int index);
	correlations_t& get_correlations() { return _correlations; }
	correlations_t const& get_correlations() const { return _correlations; }
	QList<Picviz::PVView*> process_correlation(PVView* src_view);
	void enable_correlations(bool enabled) { _correlations_enabled = enabled; }
	void remove_view_from_correlations(PVView* view);
	PVAD2GView* current_correlation() { return _current_correlation; }

public:
	void save_to_file(QString const& path, PVCore::PVSerializeArchiveOptions_p options = PVCore::PVSerializeArchiveOptions_p(), bool save_everything = false);
	void load_from_file(QString const& path);
	void load_from_archive(PVCore::PVSerializeArchive_p ar);
	PVCore::PVSerializeArchiveOptions_p get_default_serialize_options();

	void set_path(QString path) { _path = path; }
	const QString& get_path() const { return _path; }

public:
	QList<PVAD2GView_p> get_correlations_for_scene(Picviz::PVScene const& scene) const;

public:
	virtual QString get_serialize_description() const { return "Solution"; }

protected:
	bool are_correlations_serialized() const { return (bool) _so_correlations; }
	QString get_serialized_correlation_path(PVAD2GView_p const& c) const { return _so_correlations->get_child_path(c); }
protected:
	// Serialization
	void serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);
	void serialize_write(PVCore::PVSerializeObject& so);
	PVSERIALIZEOBJECT_SPLIT

private:
	PVView* _current_view = nullptr;

	correlations_t _correlations;
	PVAD2GView* _current_correlation;
	bool _correlation_running = false;
	bool _correlations_enabled = true;

	QRgb _view_colors[10] = { 0x9966CC, 0x6699CC, 0x778800, 0xFFCC66, 0x993366, 0x999999, 0x339999, 0xFF6633, 0x99FFCC, 0xFFFF99 } ;

private:
	PVCore::PVSerializeObject_p _so_correlations;
	PVCore::PVSerializeArchive_p _original_archive;
	QString _path;
};

typedef PVRoot::p_type  PVRoot_p;

}

#endif	/* PICVIZ_PVROOT_H */
