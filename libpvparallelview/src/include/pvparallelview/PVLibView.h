/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPARALLELVIEW_PVLIBVIEW_H
#define PVPARALLELVIEW_PVLIBVIEW_H

#include <sigc++/sigc++.h>

#include <inendi/PVAxesCombination.h>

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
	explicit PVLibView(Inendi::PVView& view_sp);
	~PVLibView();

  public:
	PVFullParallelView* create_view(QWidget* parent = nullptr);
	PVZoomedParallelView* create_zoomed_view(PVCombCol const axis, QWidget* parent = nullptr);
	PVHitCountView* create_hit_count_view(PVCombCol const axis, QWidget* parent = nullptr);
	PVScatterView* create_scatter_view(PVCombCol const axis, QWidget* parent = nullptr);

	void request_zoomed_zone_trees(const PVCombCol axis);
	PVZonesManager& get_zones_manager() { return _zones_manager; }
	Inendi::PVView* lib_view() { return _view; }

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
	void axes_comb_updated();
	void plotting_updated();

  private:
	Inendi::PVView* _view;
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
