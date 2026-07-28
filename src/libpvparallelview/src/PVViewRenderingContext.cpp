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

#include <squey/PVScaled.h>
#include <squey/PVView.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVViewRenderingContext.h>

PVParallelView::PVViewRenderingContext::PVViewRenderingContext(Squey::PVView& view_sp)
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
	view_sp.get_parent<Squey::PVScaled>()._scaled_updated.connect(
	    sigc::mem_fun(*this, &PVParallelView::PVViewRenderingContext::on_scaling_updated));

	view_sp._update_output_selection.connect(
	    sigc::mem_fun(*this, &PVParallelView::PVViewRenderingContext::on_selection_updated));

	view_sp._update_layer_stack_output_layer.connect(
	    sigc::mem_fun(*this, &PVParallelView::PVViewRenderingContext::on_layer_stack_output_layer_updated));

	view_sp._axis_combination_updated.connect(
	    sigc::mem_fun(*this, &PVParallelView::PVViewRenderingContext::on_axes_comb_updated));

	view_sp._axis_combination_about_to_update.connect(
	    sigc::mem_fun(*this, &PVParallelView::PVViewRenderingContext::on_axes_comb_about_to_be_updated));

	view_sp._about_to_be_delete.connect(
	    sigc::mem_fun(*this, &PVParallelView::PVViewRenderingContext::on_view_about_to_be_deleted));
}

PVParallelView::PVViewRenderingContext::~PVViewRenderingContext()
{
	PVLOG_DEBUG("In PVViewRenderingContext destructor\n");

	// Any view still attached at this point (e.g. when the whole parallel-view
	// subsystem is torn down while its widgets are still alive) can have
	// renderings in flight -- queued in the pipeline or painting on a detached
	// backend thread -- that reference this object's zones manager. Subscribed
	// views cancel and drain them, release every resource borrowed from this
	// object and detach, before the members below are destroyed (see the
	// about_to_be_deleted signal contract).
	about_to_be_deleted.emit();
}

void PVParallelView::PVViewRenderingContext::request_zoomed_zone_trees(const PVCombCol axis)
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

PVParallelView::PVZonesManager::ZoneRetainer
PVParallelView::PVViewRenderingContext::acquire_zoomed_zone(PVZoneID zone_id)
{
	PVZonesManager::ZoneRetainer zretainer = _zones_manager.acquire_zone(zone_id);
	_zones_manager.request_zoomed_zone(zone_id);

	// Keep both zones processors sized for the (possibly grown) zone count
	const size_t nzones = _zones_manager.get_number_of_zones();
	_processor_sel.reset_number_zones(nzones);
	_processor_bg.reset_number_zones(nzones);

	return zretainer;
}

void PVParallelView::PVViewRenderingContext::on_view_about_to_be_deleted()
{
	// Subscribed views drain their renderings and synchronously delete their
	// top-level widget: the model memory is released right after this handler.
	view_about_to_be_deleted.emit();

	PVParallelView::common::remove_rendering_context(*lib_view());
}

void PVParallelView::PVViewRenderingContext::on_selection_updated()
{
	// Set zones state as invalid in the according PVZonesProcessor
	for (size_t z(0); z < get_zones_manager().get_number_of_zones(); z++) {
		_processor_sel.invalidate_zone_preprocessing(get_zones_manager().get_zone_id(z));
	}

	selection_updated.emit();
}

void PVParallelView::PVViewRenderingContext::on_layer_stack_output_layer_updated()
{
	// Invalidate all background-related preprocessing
	for (size_t z(0); z < get_zones_manager().get_number_of_zones(); z++) {
		_processor_bg.invalidate_zone_preprocessing(get_zones_manager().get_zone_id(z));
		_processor_sel.invalidate_zone_preprocessing(get_zones_manager().get_zone_id(z));
	}
}

void PVParallelView::PVViewRenderingContext::on_scaling_updated(QList<PVCol> const& cols_updated)
{
	if (cols_updated.size() == 0) {
		return;
	}

	// Zones to rebuild: every alive zone (axes-combination and retained ones,
	// e.g. the off-combination zones displayed by scatter views) having one of
	// its columns updated.
	std::unordered_set<PVZoneID> zones_to_update =
	    get_zones_manager().list_zones_for_columns(cols_updated);

	zones_about_to_be_updated.emit(zones_to_update);

	for (PVZoneID z : zones_to_update) {
		get_zones_manager().update_zone(z);
		_processor_bg.invalidate_zone_preprocessing(z);
		_processor_sel.invalidate_zone_preprocessing(z);
	}

	zones_updated.emit(zones_to_update);
}

void PVParallelView::PVViewRenderingContext::on_axes_comb_about_to_be_updated()
{
	/* While the zones update, views must *not* access them: subscribed views
	 * cancel their rendering jobs and disable their widget on this signal.
	 */
	axes_combination_about_to_change.emit();
}

void PVParallelView::PVViewRenderingContext::on_axes_comb_updated(bool async /*= true*/)
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

	axes_combination_changed.emit(async);
}
