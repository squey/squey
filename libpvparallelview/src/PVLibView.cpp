//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <tbb/task_group.h>

#include <pvkernel/core/PVProgressBox.h>

#include <inendi/PVPlotted.h>
#include <inendi/PVView.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVFullParallelView.h>
#include <pvparallelview/PVFullParallelScene.h>
#include <pvparallelview/PVZoomedParallelView.h>
#include <pvparallelview/PVZoomedParallelScene.h>
#include <pvparallelview/PVHitCountView.h>
#include <pvparallelview/PVHitCountViewBackend.h>
#include <pvparallelview/PVScatterView.h>

#include <iostream>

PVParallelView::PVLibView::PVLibView(Inendi::PVView& view_sp)
    : _view(&view_sp)
    , _zones_manager(view_sp)
    , _colors(view_sp.get_output_layer_color_buffer())
    , _processor_sel(PVZonesProcessor::declare_processor_zm_sel(
          common::pipeline(), _zones_manager, _colors, view_sp.get_real_output_selection()))
    , _processor_bg(PVZonesProcessor::declare_background_processor_zm_sel(
          common::pipeline(),
          _zones_manager,
          _colors,
          view_sp.get_layer_stack_output_layer().get_selection()))
{
	view_sp.get_parent<Inendi::PVPlotted>()._plotted_updated.connect(
	    sigc::mem_fun(this, &PVParallelView::PVLibView::plotting_updated));

	view_sp._update_output_selection.connect(
	    sigc::mem_fun(this, &PVParallelView::PVLibView::selection_updated));

	view_sp._update_output_layer.connect(
	    sigc::mem_fun(this, &PVParallelView::PVLibView::output_layer_updated));

	view_sp._update_layer_stack_output_layer.connect(
	    sigc::mem_fun(this, &PVParallelView::PVLibView::layer_stack_output_layer_updated));

	view_sp._axis_combination_updated.connect(
	    sigc::mem_fun(this, &PVParallelView::PVLibView::axes_comb_updated));
	view_sp._axis_combination_about_to_update.connect(
	    sigc::mem_fun(this, &PVParallelView::PVLibView::axes_comb_about_to_be_updated));

	view_sp._about_to_be_delete.connect(
	    sigc::mem_fun(this, &PVParallelView::PVLibView::view_about_to_be_deleted));
}

PVParallelView::PVLibView::~PVLibView()
{
	PVLOG_DEBUG("In PVLibView destructor\n");
}

PVParallelView::PVFullParallelView* PVParallelView::PVLibView::create_view(QWidget* parent)
{
	PVParallelView::PVFullParallelView* view = new PVParallelView::PVFullParallelView(parent);

	PVParallelView::PVFullParallelScene* scene = new PVParallelView::PVFullParallelScene(
	    view, *lib_view(), &_sliders_manager, common::backend(), _zones_manager, _processor_sel,
	    _processor_bg);
	_parallel_scenes.push_back(scene);
	view->setScene(scene);
	scene->first_render();

	return view;
}

PVParallelView::PVZoomedParallelView*
PVParallelView::PVLibView::create_zoomed_view(PVCombCol const axis, QWidget* parent)
{
	PVParallelView::PVZoomedParallelView* view =
	    new PVParallelView::PVZoomedParallelView(lib_view()->get_axes_combination(), parent);
	PVParallelView::PVZoomedParallelScene* scene = new PVParallelView::PVZoomedParallelScene(
	    view, *lib_view(), &_sliders_manager, _processor_sel, _processor_bg, _zones_manager, axis);
	_zoomed_parallel_scenes.push_back(scene);
	view->set_scene(scene);

	return view;
}

PVParallelView::PVHitCountView* PVParallelView::PVLibView::create_hit_count_view(PVCol const axis,
                                                                                 QWidget* parent)
{
	auto create_backend = [this](PVCol axis, QWidget* parent = nullptr) {
		std::unique_ptr<PVHitCountViewBackend> backend;

		PVCore::PVProgressBox::progress(
		    [&](PVCore::PVProgressBox& pbox) {
			    pbox.set_enable_cancel(false);
			    backend = std::make_unique<PVHitCountViewBackend>(*lib_view(), axis);
			},
		    "Initializing hit-count view...", parent);
		return backend;
	};

	PVHitCountView* view = new PVHitCountView(*lib_view(), create_backend, axis, parent);

	_hit_count_views.push_back(view);

	return view;
}

PVParallelView::PVScatterView* PVParallelView::PVLibView::create_scatter_view(PVCol const axis_x,
                                                                              PVCol const axis_y,
                                                                              QWidget* parent)
{
	PVZoneID zone_id{axis_x, axis_y};

	auto create_backend = [this](PVZoneID zone_id, QWidget* parent = nullptr) {
		std::unique_ptr<PVScatterViewBackend> backend;
		PVCore::PVProgressBox::progress(
		    [&](PVCore::PVProgressBox& pbox) {
			    pbox.set_enable_cancel(false);
			    PVZonesManager::ZoneRetainer zretainer = _zones_manager.acquire_zone(zone_id);
			    _zones_manager.request_zoomed_zone(zone_id);
			    backend = std::make_unique<PVScatterViewBackend>(*lib_view(), _zones_manager,
			                                                     std::move(zretainer), zone_id,
			                                                     _processor_bg, _processor_sel);
			},
		    "Initializing scatter view...", parent);
		// Update preprocessors' number of zones
		const size_t nzones = get_zones_manager().get_number_of_zones();
		_processor_sel.reset_number_zones(nzones);
		_processor_bg.reset_number_zones(nzones);
		return backend;
	};

	PVScatterView* view = new PVScatterView(*lib_view(), create_backend, zone_id, parent);
	_scatter_views.push_back(view);

	return view;
}

void PVParallelView::PVLibView::request_zoomed_zone_trees(const PVCombCol axis)
{
	if (axis > 0) {
		_zones_manager.request_zoomed_zone(
		    PVZoneID{lib_view()->get_axes_combination().get_nraw_axis(PVCombCol(axis - 1)),
		             lib_view()->get_axes_combination().get_nraw_axis(axis)});
	}
	if (size_t(axis) < _zones_manager.get_number_of_axes_comb_zones()) {
		_zones_manager.request_zoomed_zone(
		    PVZoneID{lib_view()->get_axes_combination().get_nraw_axis(axis),
		             lib_view()->get_axes_combination().get_nraw_axis(PVCombCol(axis + 1))});
	}
}

void PVParallelView::PVLibView::view_about_to_be_deleted()
{
	for (PVFullParallelScene* view : _parallel_scenes) {
		view->about_to_be_deleted();
		delete view->graphics_view();
	}

	for (PVZoomedParallelScene* view : _zoomed_parallel_scenes) {
		view->about_to_be_deleted();
		delete view;
	}

	for (PVHitCountView* view : _hit_count_views) {
		view->about_to_be_deleted();
		delete view;
	}

	for (PVScatterView* view : _scatter_views) {
		view->about_to_be_deleted();
		delete view;
	}

	PVParallelView::common::remove_lib_view(*lib_view());
}

void PVParallelView::PVLibView::selection_updated()
{
	// Set zones state as invalid in the according PVZonesProcessor
	for (size_t z(0); z < get_zones_manager().get_number_of_zones(); z++) {
		_processor_sel.invalidate_zone_preprocessing(get_zones_manager().get_zone_id(z));
	}

	for (PVFullParallelScene* view : _parallel_scenes) {
		view->update_new_selection_async();
	}

	for (PVZoomedParallelScene* view : _zoomed_parallel_scenes) {
		view->update_new_selection_async();
	}

	for (PVHitCountView* view : _hit_count_views) {
		view->update_new_selection_async();
	}

	for (PVScatterView* view : _scatter_views) {
		view->update_new_selection_async();
	}
}

void PVParallelView::PVLibView::layer_stack_output_layer_updated()
{
	// Invalidate all background-related preprocessing
	for (size_t z(0); z < get_zones_manager().get_number_of_zones(); z++) {
		_processor_bg.invalidate_zone_preprocessing(get_zones_manager().get_zone_id(z));
		_processor_sel.invalidate_zone_preprocessing(get_zones_manager().get_zone_id(z));
	}
}
void PVParallelView::PVLibView::output_layer_updated()
{
	for (PVFullParallelScene* view : _parallel_scenes) {
		view->update_all_async();
	}

	for (PVZoomedParallelScene* view : _zoomed_parallel_scenes) {
		view->update_all_async();
	}

	for (PVHitCountView* view : _hit_count_views) {
		view->update_all_async();
	}

	for (PVScatterView* view : _scatter_views) {
		view->update_all_async();
	}
}

void PVParallelView::PVLibView::plotting_updated(QList<PVCol> const& cols_updated)
{
	if (cols_updated.size() == 0) {
		return;
	}

	// Get list of combined columns
	QSet<PVCombCol> combined_cols;
	for (PVCol col : cols_updated) {
		for (PVCombCol comb_col(0);
		     size_t(comb_col) < _zones_manager.get_number_of_axes_comb_zones() + 1; comb_col++) {
			if (lib_view()->get_axes_combination().get_nraw_axis(comb_col) == col) {
				combined_cols.insert(comb_col);
			}
		}
	}

	// Get zones from that list of columns
	std::unordered_set<PVZoneID> zones_to_update =
	    get_zones_manager().list_cols_to_zones_indices(combined_cols);

	for (PVFullParallelScene* view : _parallel_scenes) {
		view->set_enabled(false);
	}

	QList<PVZoomedParallelScene*> concerned_zoom;
	for (PVZoomedParallelScene* view : _zoomed_parallel_scenes) {
		for (PVZoneID z : zones_to_update) {
			if (view->is_zone_rendered(z)) {
				view->set_enabled(false);
				concerned_zoom.push_back(view);
				break;
			}
		}
	}

	for (PVHitCountView* view : _hit_count_views) {
		view->set_enabled(false);
	}

	QList<PVScatterView*> concerned_scatter;
	for (PVScatterView* view : _scatter_views) {
		PVZoneID zid = view->get_zone_id();
		if (cols_updated.indexOf(zid.first) != -1 or cols_updated.indexOf(zid.second) != -1) {
			view->set_enabled(false);
			concerned_scatter.push_back(view);
			zones_to_update.insert(zid);
		}
	}

	for (PVZoneID z : zones_to_update) {
		get_zones_manager().update_zone(z);
		_processor_bg.invalidate_zone_preprocessing(z);
		_processor_sel.invalidate_zone_preprocessing(z);
	}

	for (PVFullParallelScene* view : _parallel_scenes) {
		view->set_enabled(true);
		view->update_all_async();
	}

	for (PVZoomedParallelScene* view : concerned_zoom) {
		view->set_enabled(true);
		request_zoomed_zone_trees(view->get_axis_index());
		view->update_zones();
		view->update_all_async();
	}

	for (PVHitCountView* view : _hit_count_views) {
		view->set_enabled(true);
		view->update_all_async();
	}

	for (PVScatterView* view : concerned_scatter) {
		view->set_enabled(true);
		_zones_manager.request_zoomed_zone(view->get_zone_id());
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

	for (PVFullParallelScene* view : _parallel_scenes) {
		view->set_enabled(false);
	}

	for (PVZoomedParallelScene* view : _zoomed_parallel_scenes) {
		view->set_enabled(false);
	}

	for (PVHitCountView* view : _hit_count_views) {
		view->set_enabled(false);
	}

	for (PVScatterView* view : _scatter_views) {
		view->set_enabled(false);
	}
}

void PVParallelView::PVLibView::axes_comb_updated()
{
	get_zones_manager().update_from_axes_comb(*lib_view());

	// Update preprocessors' number of zones
	const size_t nzones = get_zones_manager().get_number_of_zones();
	_processor_sel.reset_number_zones(nzones);
	_processor_bg.reset_number_zones(nzones);

	// Invalidate all zones
	for (size_t z(0); z < nzones; ++z) {
		_processor_sel.invalidate_zone_preprocessing(get_zones_manager().get_zone_id(z));
		_processor_bg.invalidate_zone_preprocessing(get_zones_manager().get_zone_id(z));
	}

	for (PVFullParallelScene* view : _parallel_scenes) {
		view->set_enabled(true);
		view->update_number_of_zones();
	}

	zoomed_scene_list_t new_zps;

	for (auto & _zoomed_parallel_scene : _zoomed_parallel_scenes) {
		PVZoomedParallelScene* scene = _zoomed_parallel_scene;
		_zoomed_parallel_scene = nullptr;

		if (scene->update_zones()) {
			// the ZPS can still exist
			new_zps.push_back(scene);
		} else {
			/* the ZPS can be closed because there is no
			 * axis to rattach it to.
			 */
			scene->get_view()->parentWidget()->close();
		}
	}

	_zoomed_parallel_scenes = new_zps;

	PVCore::PVProgressBox::progress(
	    [&](PVCore::PVProgressBox& /*pbox*/) {
		    for (PVZoomedParallelScene* view : _zoomed_parallel_scenes) {
			    view->set_enabled(true);
			    request_zoomed_zone_trees(view->get_axis_index());
			    view->update_all_async();
		    }
		},
	    "Updating zoomed parallel views", nullptr);

	PVCore::PVProgressBox::progress(
	    [&](PVCore::PVProgressBox& /*pbox*/) {
		    for (PVScatterView* view : _scatter_views) {
			    view->set_enabled(true);
			    _zones_manager.request_zoomed_zone(view->get_zone_id());
			    view->update_all_async();
		    }
		},
	    "Updating scatter views", nullptr);

	PVCore::PVProgressBox::progress(
	    [&](PVCore::PVProgressBox& /*pbox*/) {
		    for (PVHitCountView* view : _hit_count_views) {
			    view->set_enabled(true);
			    view->update_all();
		    }
		},
	    "Updating hit-count views", nullptr);
}

void PVParallelView::PVLibView::remove_view(PVFullParallelScene* scene)
{
	scene_list_t::iterator it = std::find(_parallel_scenes.begin(), _parallel_scenes.end(), scene);

	if (it != _parallel_scenes.end()) {
		_parallel_scenes.erase(it);
	}
}

void PVParallelView::PVLibView::remove_zoomed_view(PVZoomedParallelScene* scene)
{
	zoomed_scene_list_t::iterator it =
	    std::find(_zoomed_parallel_scenes.begin(), _zoomed_parallel_scenes.end(), scene);

	if (it != _zoomed_parallel_scenes.end()) {
		_zoomed_parallel_scenes.erase(it);
	}
}

void PVParallelView::PVLibView::remove_hit_count_view(PVHitCountView* view)
{
	hit_count_view_list_t::iterator it =
	    std::find(_hit_count_views.begin(), _hit_count_views.end(), view);

	if (it != _hit_count_views.end()) {
		_hit_count_views.erase(it);
	}
}

void PVParallelView::PVLibView::remove_scatter_view(PVScatterView* view)
{
	scatter_view_list_t::iterator it =
	    std::find(_scatter_views.begin(), _scatter_views.end(), view);

	if (it != _scatter_views.end()) {
		_scatter_views.erase(it);
	}
}
