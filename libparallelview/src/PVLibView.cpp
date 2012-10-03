/**
 * \file PVLinesView.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <tbb/task_group.h>

#include <pvkernel/core/PVProgressBox.h>

#include <picviz/PVPlotted.h>
#include <picviz/PVView.h>

#include <pvhive/PVHive.h>
#include <pvhive/PVObserverCallback.h>
#include <pvhive/waxes/waxes.h>

#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVLibView.h>

#include <iostream>

PVParallelView::PVLibView::PVLibView(Picviz::PVView_sp& view_sp):
	_colors(view_sp->output_layer.get_lines_properties().get_buffer()),
	_zd_zt(_zones_manager, common::backend_full(), *_colors),
	_zd_zzt(_zones_manager, common::backend_zoom(), *_colors)
{
	common_init_view(view_sp);
	_zones_manager.lazy_init_from_view(*view_sp);
	common_init_zm();
}

PVParallelView::PVLibView::PVLibView(Picviz::PVView_sp& view_sp, Picviz::PVPlotted::uint_plotted_table_t const& plotted, PVRow nrows, PVCol ncols):
	_colors(view_sp->get_output_layer_color_buffer()),
	_zd_zt(_zones_manager, common::backend_full(), *_colors),
	_zd_zzt(_zones_manager, common::backend_zoom(), *_colors)
{
	common_init_view(view_sp);
	_zones_manager.set_uint_plotted(plotted, nrows, ncols);
	common_init_zm();
}

PVParallelView::PVLibView::~PVLibView()
{
	PVLOG_INFO("In PVLibView destructor\n");

}

void PVParallelView::PVLibView::common_init_view(Picviz::PVView_sp& view_sp)
{
	// Create TBB root task
	_task_root = new (tbb::task::allocate_root(_tasks_ctxt)) tbb::empty_task;
	_task_root->set_ref_count(1);

	_obs_sel = PVHive::create_observer_callback_heap<Picviz::PVSelection>(
	    [&](Picviz::PVSelection const*) { },
		[&](Picviz::PVSelection const*) { this->selection_updated(); },
		[&](Picviz::PVSelection const*) { }
	);

	_obs_output_layer = PVHive::create_observer_callback_heap<Picviz::PVLayer>(
	    [&](Picviz::PVLayer const*) { },
		[&](Picviz::PVLayer const*) { this->output_layer_updated(); },
		[&](Picviz::PVLayer const*) { }
	);

	_obs_axes_comb = PVHive::create_observer_callback_heap<Picviz::PVAxesCombination::columns_indexes_t>(
	    [&](Picviz::PVAxesCombination::columns_indexes_t const*) { },
		[&](Picviz::PVAxesCombination::columns_indexes_t const*) { this->axes_comb_updated(); },
		[&](Picviz::PVAxesCombination::columns_indexes_t const*) { }
	);

	_obs_view = PVHive::create_observer_callback_heap<Picviz::PVView>(
	    [&](Picviz::PVView const*) { },
		[&](Picviz::PVView const*) { },
		[&](Picviz::PVView const*) { this->view_about_to_be_deleted(); }
	);

	_obs_plotting = PVHive::create_observer_callback_heap<Picviz::PVPlotting>(
	    [&](Picviz::PVPlotting const*) { },
		[&](Picviz::PVPlotting const*) { this->plotting_updated(); },
		[&](Picviz::PVPlotting const*) { }
	);

	Picviz::PVPlotted_sp plotted_sp = view_sp->get_parent()->shared_from_this();

	PVHive::get().register_observer(view_sp, [=](Picviz::PVView& view) { return &view.get_real_output_selection(); }, *_obs_sel);
	PVHive::get().register_observer(view_sp, [=](Picviz::PVView& view) { return &view.get_output_layer(); }, *_obs_output_layer);
	PVHive::get().register_observer(view_sp, [=](Picviz::PVView& view) { return &view.get_axes_combination().get_axes_index_list(); }, *_obs_axes_comb);
	PVHive::get().register_observer(plotted_sp, [=](Picviz::PVPlotted& plotted) { return &plotted.get_plotting(); }, *_obs_plotting);
	PVHive::get().register_observer(view_sp, *_obs_view);

	_sliders_manager_p = PVParallelView::PVSlidersManager_p(new PVSlidersManager);
}

void PVParallelView::PVLibView::common_init_zm()
{
	_zones_manager.update_all();
}

PVParallelView::PVFullParallelView* PVParallelView::PVLibView::create_view(QWidget* parent)
{
	PVParallelView::PVFullParallelView* view = new PVParallelView::PVFullParallelView(parent);
	Picviz::PVView_sp vsp = lib_view()->shared_from_this();

	PVParallelView::PVFullParallelScene *scene = new PVParallelView::PVFullParallelScene(view, vsp, _sliders_manager_p, _zd_zt, task_root());
	_parallel_scenes.push_back(scene);
	view->setScene(scene);
	scene->first_render();

	return view;
}

PVParallelView::PVZoomedParallelView* PVParallelView::PVLibView::create_zoomed_view(PVCol const axis, QWidget* parent)
{
	PVCore::PVProgressBox pbox("Initializing zoomed parallel view");

	PVCore::PVProgressBox::progress([&]() {
			request_zoomed_zone_trees(axis);
		}, &pbox);

	PVParallelView::PVZoomedParallelView* view = new PVParallelView::PVZoomedParallelView(parent);
	Picviz::PVView_sp view_sp = lib_view()->shared_from_this();
	PVParallelView::PVZoomedParallelScene *scene = new PVParallelView::PVZoomedParallelScene(view, view_sp, _sliders_manager_p, _zd_zzt, axis);
	_zoomed_parallel_scenes.push_back(scene);
	view->setScene(scene);

	return view;
}

void PVParallelView::PVLibView::request_zoomed_zone_trees(const PVCol axis)
{
	if (axis > 0) {
		_zones_manager.request_zoomed_zone(axis - 1);
	}
	if (axis < _zones_manager.get_number_zones()) {
		_zones_manager.request_zoomed_zone(axis);
	}
}

void PVParallelView::PVLibView::view_about_to_be_deleted()
{
	for (PVFullParallelScene* view: _parallel_scenes) {
		view->about_to_be_deleted();
		view->graphics_view()->deleteLater();
	}

	for (PVZoomedParallelScene* view: _zoomed_parallel_scenes) {
		view->deleteLater();
	}

	_task_root->destroy(*_task_root);

	//PVParallelView::common::remove_lib_view(*lib_view());
}

void PVParallelView::PVLibView::selection_updated()
{
	tbb::task_group_context& ctxt = task_group_context();

	// Cancel current tasks execution
	PVLOG_INFO("PVLibView::process_selection: cancelling current tasks...\n");
	ctxt.cancel_group_execution();
	task_root()->wait_for_all();
	PVLOG_INFO("PVLibView::process_selection: tasks stopped.\n");
	ctxt.reset();

	// Flag all zones selection as invalid
	_zones_manager.invalidate_selection();

	task_root()->set_ref_count(1);

	for (PVFullParallelScene* view: _parallel_scenes) {
		view->update_new_selection();
	}

	for (PVZoomedParallelScene* view: _zoomed_parallel_scenes) {
		view->update_new_selection(task_root());
	}
}

void PVParallelView::PVLibView::output_layer_updated()
{
	for (PVFullParallelScene* view: _parallel_scenes) {
		view->update_all();
	}
}

void PVParallelView::PVLibView::plotting_updated()
{
	QList<PVCol> const& cols_updated = lib_view()->get_parent<Picviz::PVPlotted>()->last_updated_cols();
	if (cols_updated.size() == 0) {
		return;
	}

	// Get list of combined columns
	QSet<PVCol> combined_cols;
	for (PVCol c: cols_updated) {
		combined_cols.unite(lib_view()->get_axes_combination().get_combined_axes_columns_indexes(c).toSet());
	}

	// Get zones from that list of columns
	QSet<PVZoneID> zones_to_update = get_zones_manager().list_cols_to_zones(combined_cols);

	for (PVFullParallelScene* view: _parallel_scenes) {
		view->set_enabled(false);
	}

	QList<PVZoomedParallelScene*> concerned_zoom;
	for (PVZoomedParallelScene* view: _zoomed_parallel_scenes) {
		for (PVZoneID z: zones_to_update) {
			if (view->is_zone_rendered(z)) {
				view->set_enabled(false);
				concerned_zoom.push_back(view);
				break;
			}
		}
	}

	for (PVZoneID z: zones_to_update) {
		get_zones_manager().update_zone(z);
	}

	for (PVFullParallelScene* view: _parallel_scenes) {
		view->set_enabled(true);
		view->update_all();
	}

	for (PVZoomedParallelScene* view: concerned_zoom) {
		view->set_enabled(true);
		request_zoomed_zone_trees(view->get_axis_index());
		view->update_all();
	}
}

void PVParallelView::PVLibView::axes_comb_updated()
{
	/* while the zones update, views must *not* access to them;
	 * views have also to be disabled (jobs must be cancelled
	 * and the widgets must be disabled in the Qt's way).
	 */
	// FIXME: the running selection update must be stopped too.

	for (PVFullParallelScene* view: _parallel_scenes) {
		view->set_enabled(false);
	}

	for (PVZoomedParallelScene* view: _zoomed_parallel_scenes) {
		view->set_enabled(false);
	}

	get_zones_manager().update_from_axes_comb(*lib_view());

	for (PVFullParallelScene* view: _parallel_scenes) {
		view->set_enabled(true);
		view->update_number_of_zones();
	}

	for (PVZoomedParallelScene* view: _zoomed_parallel_scenes) {
		view->set_enabled(true);
		// FIXME: if ::update_zones return false, the view
		// can be deleted.
		view->update_zones();
	}

	PVCore::PVProgressBox pbox("Reinitializing zoomed parallel views");

	PVCore::PVProgressBox::progress([&]() {
			for (PVZoomedParallelScene* view: _zoomed_parallel_scenes) {
				request_zoomed_zone_trees(view->get_axis_index());
				view->update_all();
			}
		}, &pbox);
}

void PVParallelView::PVLibView::remove_view(PVFullParallelScene *scene)
{
	scene_list_t::iterator it = std::find(_parallel_scenes.begin(),
	                                      _parallel_scenes.end(),
	                                      scene);

	if (it != _parallel_scenes.end()) {
		_parallel_scenes.erase(it);
	}
}

void PVParallelView::PVLibView::remove_zoomed_view(PVZoomedParallelScene *scene)
{
	zoomed_scene_list_t::iterator it = std::find(_zoomed_parallel_scenes.begin(),
	                                             _zoomed_parallel_scenes.end(),
	                                             scene);

	if (it != _zoomed_parallel_scenes.end()) {
		_zoomed_parallel_scenes.erase(it);
	}
}
