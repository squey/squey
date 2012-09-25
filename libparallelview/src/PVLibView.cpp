/**
 * \file PVLinesView.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <tbb/task_group.h>

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
	for (PVFullParallelScene& view: _parallel_scenes) {
		view.about_to_be_deleted();
		view.graphics_view()->deleteLater();
	}

	for (PVZoomedParallelScene& view: _zoomed_parallel_scenes) {
		view.deleteLater();
	}

	_task_root->destroy(*_task_root);
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

	PVHive::get().register_observer(view_sp, [=](Picviz::PVView& view) { return &view.get_real_output_selection(); }, *_obs_sel);
	PVHive::get().register_observer(view_sp, [=](Picviz::PVView& view) { return &view.get_output_layer(); }, *_obs_output_layer);
	PVHive::get().register_observer(view_sp, [=](Picviz::PVView& view) { return &view.get_axes_combination().get_axes_index_list(); }, *_obs_axes_comb);
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
	_parallel_scenes.emplace_back(view, vsp, _sliders_manager_p, _zd_zt, task_root());
	PVFullParallelScene& scene = _parallel_scenes.back();
	view->setScene(&scene);
	scene.first_render();
	return view;
}

PVParallelView::PVZoomedParallelView* PVParallelView::PVLibView::create_zoomed_view(PVCol const axis, QWidget* parent)
{
	PVParallelView::PVZoomedParallelView* view = new PVParallelView::PVZoomedParallelView(parent);
	Picviz::PVView_sp view_sp = lib_view()->shared_from_this();
	_zoomed_parallel_scenes.emplace_back(view, view_sp, _sliders_manager_p, _zd_zzt, axis);
	view->setScene(&_zoomed_parallel_scenes.back());

	return view;
}
void PVParallelView::PVLibView::view_about_to_be_deleted()
{
	PVParallelView::common::remove_lib_view(*lib_view());
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

	for (PVFullParallelScene& view: _parallel_scenes) {
		view.update_new_selection();
	}

	for (PVZoomedParallelScene& view: _zoomed_parallel_scenes) {
		view.update_new_selection(task_root());
	}
}

void PVParallelView::PVLibView::output_layer_updated()
{
	for (PVFullParallelScene& view: _parallel_scenes) {
		view.update_all();
	}
}

void PVParallelView::PVLibView::axes_comb_updated()
{
	/* while the zones update, views must *not* access to them;
	 * views have also to be disabled (jobs must be cancelled
	 * and the widgets must be disabled in the Qt's way).
	 */
	// FIXME: the running selection update must be stopped too.

	for (PVFullParallelScene& view: _parallel_scenes) {
		view.set_enabled(false);
	}

	for (PVZoomedParallelScene& view: _zoomed_parallel_scenes) {
		view.set_enabled(false);
	}

	get_zones_manager().update_from_axes_comb(*lib_view());

	for (PVFullParallelScene& view: _parallel_scenes) {
		view.set_enabled(true);
		view.update_number_of_zones();
	}

	for (PVZoomedParallelScene& view: _zoomed_parallel_scenes) {
		view.set_enabled(true);
		// FIXME: if ::update_zones return false, the view
		// can be deleted.
		view.update_zones();
	}
}
