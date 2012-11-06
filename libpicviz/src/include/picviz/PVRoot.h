/**
 * \file PVRoot.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PICVIZ_PVROOT_H
#define PICVIZ_PVROOT_H

#include <QList>
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
	PVAD2GView_p get_correlation(int index);
	void select_correlation(int index) { if (index == -1) _current_correlation.reset(); else _current_correlation = get_correlation(index); }
	void add_correlation(const QString & name);
	void delete_correlation(int index);
	correlations_t& get_correlations() { return _correlations; }
	QList<Picviz::PVView*> process_correlation(PVView* src_view);

public:
	virtual QString get_serialize_description() const { return "Root"; }

private:
	// Plugins loading
	static int load_layer_filters();
	static int load_mapping_filters();
	static int load_plotting_filters();
	static int load_row_filters();
	static int load_axis_computation_filters();
	static int load_sorting_functions_filters();

private:
	static PVRoot_sp _unique_root;
	correlations_t _correlations;
	PVAD2GView_p _current_correlation;
	bool _correlation_running = false;
};

typedef PVRoot::p_type  PVRoot_p;

}

#endif	/* PICVIZ_PVROOT_H */
