/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVPARALLELVIEW_PVVIEWRENDERINGCONTEXT_H
#define PVPARALLELVIEW_PVVIEWRENDERINGCONTEXT_H

#include <sigc++/sigc++.h>

#include <squey/PVAxesCombination.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVZonesProcessor.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVSlidersManager.h>

#include <unordered_set>

namespace PVParallelView
{

class PVZonesManager;

/**
 * Rendering resources shared by every graphical view of one Squey::PVView:
 * the zones manager (zone trees are expensive to build), the sliders manager
 * and the two zones processors bound to the global rendering pipeline.
 *
 * One instance per Squey::PVView, created lazily through
 * common::get_rendering_context() and destroyed when the model view dies (or
 * at backend teardown). This class knows no concrete view type: views are
 * created by the PVDisplays plugins and react to the model through the sigc
 * signals below, which guarantee that the shared state has been invalidated
 * before subscribers run.
 */
class PVViewRenderingContext : public sigc::trackable
{
  public:
	explicit PVViewRenderingContext(Squey::PVView& view_sp);
	~PVViewRenderingContext();

  public:
	void request_zoomed_zone_trees(const PVCombCol axis);

	/**
	 * Acquire (building it if needed) an off-combination zone together with
	 * its zoomed zone tree, keeping both zones processors sized for the new
	 * zone count. Used by scatter views.
	 */
	PVZonesManager::ZoneRetainer acquire_zoomed_zone(PVZoneID zone_id);

	PVZonesManager& get_zones_manager() { return _zones_manager; }
	PVSlidersManager& sliders_manager() { return _sliders_manager; }
	PVZonesProcessor& processor_sel() { return _processor_sel; }
	PVZonesProcessor& processor_bg() { return _processor_bg; }
	Squey::PVView* lib_view() { return _view; }

  public:
	/* Signals relayed from the Squey::PVView model to the attached views.
	 *
	 * Connection rule: any event whose semantics include "the zones manager /
	 * zones processors state is up to date" MUST be observed through these
	 * signals (this object subscribes to the model first, performs its
	 * internal invalidations, then emits). Events that do not depend on the
	 * shared rendering state (e.g. PVView::_update_output_layer,
	 * _toggle_unselected_zombie_visibility) can be observed directly on the
	 * model. Views inherit sigc::trackable, so disconnection is automatic.
	 */

	/* Emitted after the selection preprocessing of every zone has been
	 * invalidated in the selection processor. Subscribers typically schedule
	 * an asynchronous redraw of their selection layer. */
	sigc::signal<void()> selection_updated;

	/* Two-phase protocol around an axes combination change. Between the two
	 * signals the zones manager is rebuilt and both processors are resized
	 * and fully invalidated; subscribers must cancel their renderings and
	 * stop accessing zones on the first signal, and may resume on the second.
	 * The 'async' flag mirrors Squey::PVView::_axis_combination_updated:
	 * when false, subscribers must update synchronously and must not open
	 * progress dialogs. */
	sigc::signal<void()> axes_combination_about_to_change;
	sigc::signal<void(bool /*async*/)> axes_combination_changed;

	/* Two-phase protocol around in-place zone rebuilds (scaling updates).
	 * The payload lists every impacted zone (axes-combination and retained
	 * ones). Between the two signals each listed zone is rebuilt and
	 * invalidated in both processors. */
	sigc::signal<void(std::unordered_set<PVZoneID> const&)> zones_about_to_be_updated;
	sigc::signal<void(std::unordered_set<PVZoneID> const&)> zones_updated;

	/* The Squey::PVView is being destroyed. Subscribers must cancel their
	 * renderings and synchronously delete their top-level widget: the model
	 * memory is released right after this emission. */
	sigc::signal<void()> view_about_to_be_deleted;

	/* This object itself is being destroyed while some views are still alive
	 * (e.g. backend teardown before model teardown). Subscribers must
	 * synchronously cancel their renderings, release every resource borrowed
	 * from this object (zone retainers, sliders registrations) and detach:
	 * after this emission any use of this object is undefined. Widgets are
	 * NOT deleted here; they stay alive, inert, until their Qt parent
	 * destroys them. */
	sigc::signal<void()> about_to_be_deleted;

  protected:
	void on_selection_updated();
	void on_selection_view_changed();
	void on_layer_stack_output_layer_updated();
	void on_view_about_to_be_deleted();
	void on_axes_comb_about_to_be_updated();
	void on_axes_comb_updated(bool async = true);
	void on_scaling_updated(QList<PVCol> const& cols_updated);

  private:
	Squey::PVView* _view;
	PVZonesManager _zones_manager;
	PVSlidersManager _sliders_manager;
	PVCore::PVHSVColor const* _colors;

	PVZonesProcessor _processor_sel;
	PVZonesProcessor _processor_bg;
};
} // namespace PVParallelView

#endif /* PVPARALLELVIEW_PVVIEWRENDERINGCONTEXT_H */
