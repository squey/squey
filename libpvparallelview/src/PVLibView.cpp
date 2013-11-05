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

#include <pvparallelview/common.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVFullParallelView.h>
#include <pvparallelview/PVFullParallelScene.h>
#include <pvparallelview/PVZoomedParallelView.h>
#include <pvparallelview/PVZoomedParallelScene.h>
#include <pvparallelview/PVHitCountView.h>
#include <pvparallelview/PVScatterView.h>

#include <iostream>

PVParallelView::PVLibView::PVLibView(Picviz::PVView_sp& view_sp):
	_colors(view_sp->output_layer.get_lines_properties().get_buffer())
{
	common_init_view(view_sp);
	_zones_manager.lazy_init_from_view(*view_sp);
	common_init_zm();
}

PVParallelView::PVLibView::PVLibView(Picviz::PVView_sp& view_sp, Picviz::PVPlotted::uint_plotted_table_t const& plotted, PVRow nrows, PVCol ncols):
	_colors(view_sp->get_output_layer_color_buffer())
{
	common_init_view(view_sp);
	_zones_manager.set_uint_plotted(plotted, nrows, ncols);
	common_init_zm();
}

PVParallelView::PVLibView::~PVLibView()
{
	PVLOG_DEBUG("In PVLibView destructor\n");
}

void PVParallelView::PVLibView::common_init_view(Picviz::PVView_sp& view_sp)
{
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

	_obs_layer_stack_output_layer = PVHive::create_observer_callback_heap<Picviz::PVLayer>(
	    [&](Picviz::PVLayer const*) { },
		[&](Picviz::PVLayer const*) { this->layer_stack_output_layer_updated(); },
		[&](Picviz::PVLayer const*) { }
	);

	_obs_axes_comb = PVHive::create_observer_callback_heap<Picviz::PVAxesCombination::columns_indexes_t>(
	    [&](Picviz::PVAxesCombination::columns_indexes_t const*) { this->axes_comb_about_to_be_updated(); },
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


	PVHive::get().register_observer(view_sp, [=](Picviz::PVView& view) { return &view.get_real_output_selection(); }, *_obs_sel);
	PVHive::get().register_observer(view_sp, [=](Picviz::PVView& view) { return &view.get_output_layer(); }, *_obs_output_layer);
	PVHive::get().register_observer(view_sp, [=](Picviz::PVView& view) { return &view.get_layer_stack_output_layer(); }, *_obs_layer_stack_output_layer);
	PVHive::get().register_observer(view_sp, [=](Picviz::PVView& view) { return &view.get_axes_combination().get_axes_index_list(); }, *_obs_axes_comb);
	PVHive::get().register_observer(view_sp, *_obs_view);

	if (view_sp->get_parent()) {
		Picviz::PVPlotted_sp plotted_sp = view_sp->get_parent()->shared_from_this();
		PVHive::get().register_observer(plotted_sp, [=](Picviz::PVPlotted& plotted) { return &plotted.get_plotting(); }, *_obs_plotting);
	}

	_sliders_manager_p = PVParallelView::PVSlidersManager_p(new PVSlidersManager);
}

void PVParallelView::PVLibView::common_init_zm()
{
	_zones_manager.update_all();

	PVCore::PVHSVColor const* const colors = lib_view()->get_output_layer().get_lines_properties().get_buffer();

	// Init zones processors
	_processor_sel = PVZonesProcessor::declare_processor_zm_sel(common::pipeline(), _zones_manager,
		colors,
		lib_view()->get_real_output_selection());

	_processor_bg = PVZonesProcessor::declare_background_processor_zm_sel(common::pipeline(), _zones_manager,
		colors,
		lib_view()->get_layer_stack_output_layer().get_selection());
}

PVParallelView::PVFullParallelView* PVParallelView::PVLibView::create_view(QWidget* parent)
{
	PVParallelView::PVFullParallelView* view = new PVParallelView::PVFullParallelView(parent);
	Picviz::PVView_sp vsp = lib_view()->shared_from_this();

	PVParallelView::PVFullParallelScene *scene = new PVParallelView::PVFullParallelScene(view, vsp, _sliders_manager_p, common::backend(), _zones_manager, _processor_sel, _processor_bg);
	_parallel_scenes.push_back(scene);
	view->setScene(scene);
	scene->first_render();

	return view;
}

PVParallelView::PVZoomedParallelView* PVParallelView::PVLibView::create_zoomed_view(PVCol const axis, QWidget* parent)
{
	PVCore::PVProgressBox pbox("Initializing zoomed parallel view");

	pbox.set_enable_cancel(false);

	PVCore::PVProgressBox::progress([&]() {
			request_zoomed_zone_trees(axis);
		}, &pbox);

	PVParallelView::PVZoomedParallelView* view = new PVParallelView::PVZoomedParallelView(parent);
	Picviz::PVView_sp view_sp = lib_view()->shared_from_this();
	PVParallelView::PVZoomedParallelScene *scene = new PVParallelView::PVZoomedParallelScene(view, view_sp, _sliders_manager_p, _processor_sel, _processor_bg, _zones_manager, axis);
	_zoomed_parallel_scenes.push_back(scene);
	view->set_scene(scene);

	return view;
}

PVParallelView::PVHitCountView* PVParallelView::PVLibView::create_hit_count_view(PVCol const axis,
	                                                                         QWidget* parent)
{
	Picviz::PVView_sp view_sp = lib_view()->shared_from_this();

	const uint32_t *uint_plotted =
		Picviz::PVPlotted::get_plotted_col_addr(_zones_manager.get_uint_plotted(),
				_zones_manager.get_number_rows(),
				lib_view()->get_original_axis_index(axis));

	PVHitCountView* view = new PVHitCountView(view_sp,
	                                          uint_plotted,
	                                          _zones_manager.get_number_rows(),
	                                          axis,
	                                          parent);

	_hit_count_views.push_back(view);

	return view;
}


PVParallelView::PVScatterView* PVParallelView::PVLibView::create_scatter_view(
	const PVCol axis,
	QWidget* parent
)
{
	Picviz::PVView_sp view_sp = lib_view()->shared_from_this();
	_zones_manager.request_zoomed_zone(axis);

	PVScatterView* view = new PVScatterView(
		view_sp,
        _zones_manager,
		axis,
		_processor_bg,
		_processor_sel,
        parent
    );

	_scatter_views.push_back(view);

	return view;
}

void PVParallelView::PVLibView::request_zoomed_zone_trees(const PVCol axis)
{
	if (axis > 0) {
		_zones_manager.request_zoomed_zone(axis - 1);
	}
	if (axis < _zones_manager.get_number_of_managed_zones()) {
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
		view->about_to_be_deleted();
		view->deleteLater();
	}

	for (PVHitCountView* view: _hit_count_views) {
		view->about_to_be_deleted();
		view->deleteLater();
	}

	for (PVScatterView* view: _scatter_views) {
		view->about_to_be_deleted();
		view->deleteLater();
	}

	PVParallelView::common::remove_lib_view(*lib_view());
}

void PVParallelView::PVLibView::selection_updated()
{
	// Set zones state as invalid in the according PVZonesProcessor
	for (PVZoneID z = 0; z < get_zones_manager().get_number_of_managed_zones(); z++) {
		_processor_sel.invalidate_zone_preprocessing(z);
	}

	for (PVFullParallelScene* view: _parallel_scenes) {
		view->update_new_selection_async();
	}

	for (PVZoomedParallelScene* view: _zoomed_parallel_scenes) {
		view->update_new_selection_async();
	}

	for (PVHitCountView* view: _hit_count_views) {
		view->update_new_selection_async();
	}

	for (PVScatterView* view: _scatter_views) {
		view->update_new_selection_async();
	}
}

void PVParallelView::PVLibView::layer_stack_output_layer_updated()
{
	// Invalidate all background-related preprocessing
	for (PVZoneID z = 0; z < get_zones_manager().get_number_of_managed_zones(); z++) {
		_processor_bg.invalidate_zone_preprocessing(z);
		_processor_sel.invalidate_zone_preprocessing(z);
	}
}
void PVParallelView::PVLibView::output_layer_updated()
{
	for (PVFullParallelScene* view: _parallel_scenes) {
		view->update_all_async();
	}

	for (PVZoomedParallelScene* view: _zoomed_parallel_scenes) {
		view->update_all_async();
	}

	for (PVHitCountView* view: _hit_count_views) {
		view->update_all_async();
	}

	for (PVScatterView* view: _scatter_views) {
		view->update_all_async();
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

	for (PVHitCountView* view: _hit_count_views) {
		view->set_enabled(false);
	}

	QList<PVScatterView*> concerned_scatter;
	for (PVScatterView* view: _scatter_views) {
		for (PVZoneID z: zones_to_update) {
			if (view->get_zone_index() == z) {
				view->set_enabled(false);
				concerned_scatter.push_back(view);
				break;
			}
		}
	}

	for (PVZoneID z: zones_to_update) {
		get_zones_manager().update_zone(z);
		_processor_bg.invalidate_zone_preprocessing(z);
		_processor_sel.invalidate_zone_preprocessing(z);
	}

	for (PVFullParallelScene* view: _parallel_scenes) {
		view->set_enabled(true);
		view->update_all_async();
	}

	for (PVZoomedParallelScene* view: concerned_zoom) {
		view->set_enabled(true);
		request_zoomed_zone_trees(view->get_axis_index());
		view->update_zones();
		view->update_all_async();
	}

	for (PVHitCountView* view: _hit_count_views) {
		view->set_enabled(true);
		view->update_all_async();
	}

	for (PVScatterView* view: concerned_scatter) {
		view->set_enabled(true);
		request_zoomed_zone_trees(view->get_zone_index());
		view->update_zones();
		view->update_all_async();
	}
}

void PVParallelView::PVLibView::axes_comb_about_to_be_updated()
{
	/* while the zones update, views must *not* access to them;
	 * views have also to be disabled (jobs must be cancelled
	 * and the widgets must be disabled in the Qt's way).
	 */

	for (PVFullParallelScene* view: _parallel_scenes) {
		view->set_enabled(false);
	}

	for (PVZoomedParallelScene* view: _zoomed_parallel_scenes) {
		view->set_enabled(false);
	}

	for (PVHitCountView* view: _hit_count_views) {
		view->set_enabled(false);
	}

	for (PVScatterView* view: _scatter_views) {
		view->set_enabled(false);
	}
}

void PVParallelView::PVLibView::axes_comb_updated()
{
	std::vector<PVZoneID> modified_zones(get_zones_manager().update_from_axes_comb(*lib_view()));

	// Update preprocessors' number of zones
	const PVZoneID nzones = get_zones_manager().get_number_of_managed_zones();
	_processor_sel.set_number_zones(nzones);
	_processor_bg.set_number_zones(nzones);

	// Invalidate modified zones
	for (PVZoneID const z: modified_zones) {
		_processor_sel.invalidate_zone_preprocessing(z);
		_processor_bg.invalidate_zone_preprocessing(z);
	}

	for (PVFullParallelScene* view: _parallel_scenes) {
		view->set_enabled(true);
		view->update_number_of_zones_async();
	}

	zoomed_scene_list_t new_zps;

	for (size_t i = 0; i < _zoomed_parallel_scenes.size(); ++i) {
		PVZoomedParallelScene* scene = _zoomed_parallel_scenes[i];
		_zoomed_parallel_scenes[i] = nullptr;

		if (scene->update_zones()) {
			// the ZPS can still exist
			new_zps.push_back(scene);
		} else {
			/* the ZPS can be closed because there is no
			 * axis to rattach it to.
			 */
			for(QGraphicsView *view: scene->views()) {
				if (view->parentWidget() == nullptr) {
					PVLOG_WARN("a ZoomedParallelScene exists but is not in a dock!\n");
					continue;
				}
				view->parentWidget()->close();
			}
		}
	}

	_zoomed_parallel_scenes = new_zps;

	scatter_view_list_t new_svs;

	for (size_t i = 0; i < _scatter_views.size(); ++i) {
		PVScatterView* sv = _scatter_views[i];
		_scatter_views[i] = nullptr;

		if (sv->update_zones()) {
			// the ZPS can still exist
			new_svs.push_back(sv);
		} else {
			sv->parentWidget()->close();
		}
	}

	_scatter_views = new_svs;

	hit_count_view_list_t new_hcvs;

	for (size_t i = 0; i < _hit_count_views.size(); ++i) {
		PVHitCountView* hcv = _hit_count_views[i];
		_hit_count_views[i] = nullptr;

		if (hcv->update_zones()) {
			// the ZPS can still exist
			new_hcvs.push_back(hcv);
		} else {
			hcv->parentWidget()->close();
		}
	}

	_hit_count_views = new_hcvs;

	PVCore::PVProgressBox pbox("Updating zoomed parallel views");

	PVCore::PVProgressBox::progress([&]() {
			for (PVZoomedParallelScene* view: _zoomed_parallel_scenes) {
				view->set_enabled(true);
				request_zoomed_zone_trees(view->get_axis_index());
				view->update_all_async();
			}
		}, &pbox);

	PVCore::PVProgressBox pbox2("Updating scatter views");

	PVCore::PVProgressBox::progress([&]() {
			for (PVScatterView* view: _scatter_views) {
				view->set_enabled(true);
				request_zoomed_zone_trees(view->get_zone_index());
				view->update_all_async();
			}
		}, &pbox2);

	PVCore::PVProgressBox pbox3("Updating hit-count views");

	PVCore::PVProgressBox::progress([&]() {
			for (PVHitCountView* view: _hit_count_views) {
				view->set_enabled(true);
				view->update_all_async();
			}
		}, &pbox3);
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

void PVParallelView::PVLibView::remove_hit_count_view(PVHitCountView *view)
{
	hit_count_view_list_t::iterator it = std::find(_hit_count_views.begin(),
	                                               _hit_count_views.end(),
	                                               view);

	if (it != _hit_count_views.end()) {
		_hit_count_views.erase(it);
	}
}

void PVParallelView::PVLibView::remove_scatter_view(PVScatterView *view)
{
	scatter_view_list_t::iterator it = std::find(
		_scatter_views.begin(),
		_scatter_views.end(),
	    view
	);

	if (it != _scatter_views.end()) {
		_scatter_views.erase(it);
	}
}
