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
#include <picviz/PVPtrObjects.h> // For PVScene_p
#include <pvkernel/core/PVDataTreeObject.h>

#include <boost/shared_ptr.hpp>

#include <picviz/PVRoot_types.h>
#include <picviz/PVAD2GView.h>

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
	friend class PVSource;
	//typedef boost::shared_ptr<PVRoot> p_type;
	typedef std::list<PVAD2GView_p> correlations_t;

private:
	PVRoot();

public:
	~PVRoot();

public:
	static PVRoot& get_root(); 
	static PVRoot_sp get_root_sp();
	static void release();

public:
	PVView* current_view();
	PVView const* current_view() const;
	void select_view(PVView& view);

public:
	int32_t get_new_view_id();
	void set_views_id();
	QColor get_new_view_color();

public:
	PVAD2GView_p get_correlation(int index);
	void select_correlation(int index) { if (index == -1) _current_correlation.reset(); else _current_correlation = get_correlation(index); }
	void add_correlation(const QString & name);
	void delete_correlation(int index);
	correlations_t& get_correlations() { return _correlations; }
	QList<Picviz::PVView*> process_correlation(PVView* src_view);
	void enable_correlations(bool enabled) { _correlations_enabled = enabled; }
	void remove_view_from_correlations(PVView* view);

public:
	virtual QString get_serialize_description() const { return "Root"; }

private:
	static PVRoot_sp _unique_root;

	PVView* _current_view = nullptr;
	int _new_view_id = 0;

	correlations_t _correlations;
	PVAD2GView_p _current_correlation;
	bool _correlation_running = false;
	bool _correlations_enabled = true;

	QList<QRgb> _available_colors;
	QList<QRgb> _used_colors;
};

typedef PVRoot::p_type  PVRoot_p;

}

#endif	/* PICVIZ_PVROOT_H */
