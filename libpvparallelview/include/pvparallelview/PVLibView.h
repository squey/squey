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

#ifndef PVPARALLELVIEW_PVLIBVIEW_H
#define PVPARALLELVIEW_PVLIBVIEW_H

#include <sigc++/sigc++.h>

#include <squey/PVAxesCombination.h>

#include <pvparallelview/PVZonesProcessor.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVSlidersManager.h>

namespace PVParallelView
{

class PVZonesManager;
class PVFuncObserverSignal;
class PVFullParallelScene;
class PVFullParallelView;
class PVZoomedParallelScene;
class PVZoomedParallelView;
class PVHitCountView;
class PVScatterView;

class PVLibView : public sigc::trackable
{
  private:
	typedef std::list<PVFullParallelScene*> scene_list_t;
	typedef std::vector<PVZoomedParallelScene*> zoomed_scene_list_t;
	typedef std::vector<PVHitCountView*> hit_count_view_list_t;
	typedef std::vector<PVScatterView*> scatter_view_list_t;

  public:
	explicit PVLibView(Squey::PVView& view_sp);
	~PVLibView();

  public:
	PVFullParallelView* create_view(QWidget* parent = nullptr);
	PVZoomedParallelView* create_zoomed_view(PVCombCol const axis, QWidget* parent = nullptr);
	PVHitCountView* create_hit_count_view(PVCol const axis, QWidget* parent = nullptr);
	PVScatterView*
	create_scatter_view(PVCol const axis_x, PVCol const axis_y, QWidget* parent = nullptr);

	void request_zoomed_zone_trees(const PVCombCol axis);
	PVZonesManager& get_zones_manager() { return _zones_manager; }
	Squey::PVView* lib_view() { return _view; }

	void remove_view(PVFullParallelScene* scene);
	void remove_zoomed_view(PVZoomedParallelScene* scene);
	void remove_hit_count_view(PVHitCountView* view);
	void remove_scatter_view(PVScatterView* view);

  protected:
	void selection_updated();
	void output_layer_updated();
	void layer_stack_output_layer_updated();
	void view_about_to_be_deleted();
	void axes_comb_about_to_be_updated();
	void axes_comb_updated(bool async = true);
	void scaling_updated(QList<PVCol> const& cols_updated);

  private:
	Squey::PVView* _view;
	PVZonesManager _zones_manager;
	PVSlidersManager _sliders_manager;
	scene_list_t _parallel_scenes;
	zoomed_scene_list_t _zoomed_parallel_scenes;
	hit_count_view_list_t _hit_count_views;
	scatter_view_list_t _scatter_views;
	PVCore::PVHSVColor const* _colors;

	PVZonesProcessor _processor_sel;
	PVZonesProcessor _processor_bg;
};
} // namespace PVParallelView

#endif /* PVPARALLELVIEW_PVLIBVIEW_H */
