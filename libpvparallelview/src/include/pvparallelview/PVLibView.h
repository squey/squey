/**
 * \file PVLibView.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_PVLIBVIEW_H
#define PVPARALLELVIEW_PVLIBVIEW_H

#include <picviz/PVAxesCombination.h>
#include <picviz/PVPlotting.h>
#include <picviz/PVView_types.h>

#include <pvhive/PVObserverSignal.h>
#include <pvhive/PVCallHelper.h>

#include <pvparallelview/PVFullParallelScene.h>
#include <pvparallelview/PVZoomedParallelScene.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVSlidersManager.h>

#include <QObject>

#include <tbb/task.h>

namespace PVParallelView
{

class PVZonesManager;
class PVFuncObserverSignal;

class PVLibView
{
private:
	typedef std::list<PVFullParallelScene*> scene_list_t;
	typedef std::vector<PVZoomedParallelScene*> zoomed_scene_list_t;
	friend class process_selection_Observer;

public:
	PVLibView(Picviz::PVView_sp& view_sp);
	// For testing purposes
	PVLibView(Picviz::PVView_sp& view_sp, Picviz::PVPlotted::uint_plotted_table_t const& plotted, PVRow nrows, PVCol ncols);
	~PVLibView();

public:
	PVFullParallelView* create_view(QWidget* parent = NULL);
	PVZoomedParallelView* create_zoomed_view(PVCol const axis, QWidget* parent = NULL);
	void request_zoomed_zone_trees(const PVCol axis);
	PVZonesManager& get_zones_manager() { return _zones_manager; }
	Picviz::PVView* lib_view() { return _obs_view->get_object(); }

	void remove_view(PVFullParallelScene *scene);
	void remove_zoomed_view(PVZoomedParallelScene *scene);

protected:
	void selection_updated();
	void output_layer_updated();
	void view_about_to_be_deleted();
	void axes_comb_updated();
	void plotting_updated();

protected:
	void common_init_view(Picviz::PVView_sp& view_sp);
	void common_init_zm();
	tbb::task* task_root() { return _task_root; }
	tbb::task_group_context& task_group_context() { return _tasks_ctxt; }

private:
	PVZonesManager                            _zones_manager;
	PVSlidersManager_p                        _sliders_manager_p;
	PVHive::PVObserver_p<Picviz::PVLayer>     _obs_output_layer;
	PVHive::PVObserver_p<Picviz::PVSelection> _obs_sel;
	PVHive::PVObserver_p<Picviz::PVView>      _obs_view;
	PVHive::PVObserver_p<Picviz::PVAxesCombination::columns_indexes_t> _obs_axes_comb;
	PVHive::PVObserver_p<Picviz::PVPlotting>  _obs_plotting;
	scene_list_t                              _parallel_scenes;
	zoomed_scene_list_t                       _zoomed_parallel_scenes;
	PVCore::PVHSVColor                 const* _colors;
	tbb::task                                *_task_root;
	tbb::task_group_context                   _tasks_ctxt;

	PVZonesDrawing<PARALLELVIEW_ZT_BBITS>     _zd_zt;
	PVZonesDrawing<PARALLELVIEW_ZZT_BBITS>    _zd_zzt;
};

}

#endif /* PVPARALLELVIEW_PVLIBVIEW_H */