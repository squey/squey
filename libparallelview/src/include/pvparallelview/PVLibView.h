/**
 * \file PVLibView.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_PVLIBVIEW_H
#define PVPARALLELVIEW_PVLIBVIEW_H

#include <picviz/FakePVView.h>

#include <pvhive/PVFuncObserver.h>
#include <pvhive/PVCallHelper.h>

#include <pvparallelview/PVFullParallelScene.h>
#include <pvparallelview/PVZoomedParallelScene.h>
#include <pvparallelview/PVZonesManager.h>

#include <pvparallelview/PVBCIDrawingBackendCUDA.h>

#include <tbb/task.h>

namespace PVParallelView
{

class PVZonesManager;
class PVFuncObserverSignal;

class PVLibView
{
private:
	typedef std::list<PVFullParallelScene> views_list_t;
	typedef std::list<PVZoomedParallelScene> zoomed_scene_list_t;
	friend class process_selection_Observer;

public:
	PVLibView(Picviz::FakePVView::shared_pointer& view_sp, PVCore::PVHSVColor *colors);
	~PVLibView();

public:
	void create_view(PVParallelView::PVLinesView::zones_drawing_t::bci_backend_t& bci_backend)
	{
		_parallel_views.emplace_back(_view_sp, _zones_manager, bci_backend, task_root());
		_parallel_views.back().first_render();
	}

	template <typename Backend>
	void create_zoomed_scene(PVParallelView::PVZoomedParallelView *zpv,
	                         PVCol axis)
	{
		Backend &zoom_backend = *(new Backend);
		PVParallelView::PVZoomedParallelScene::zones_drawing_t &zzd =
			*(new PVParallelView::PVZoomedParallelScene::zones_drawing_t(_zones_manager,
			                                                             zoom_backend,
			                                                             *_colors));
		_zoomed_parallel_scenes.emplace_back(zpv, _view_sp, zzd, axis);
		zpv->setScene(&_zoomed_parallel_scenes.back());
	}

	PVZonesManager& get_zones_manager() { return _zones_manager; }

protected:
	tbb::task* task_root() { return _task_root; }
	tbb::task_group_context& task_group_context() { return _tasks_ctxt; }

private:
	class process_selection_Observer: public PVHive::PVFuncObserverSignal<Picviz::FakePVView, FUNC(Picviz::FakePVView::process_selection)>
	{
		public:
			process_selection_Observer(PVLibView* parent) : _parent(parent) {}
		protected:
			virtual void update(arguments_deep_copy_type const& args) const;
		private:
			PVLibView* _parent;
	};

private:
	PVZonesManager                      _zones_manager;
	process_selection_Observer         *_process_selection_observer;
	views_list_t                        _parallel_views;
	zoomed_scene_list_t                 _zoomed_parallel_scenes;
	Picviz::FakePVView::shared_pointer &_view_sp;
	PVCore::PVHSVColor                 *_colors;
	tbb::task                          *_task_root;
	tbb::task_group_context             _tasks_ctxt;

};

}

#endif /* PVPARALLELVIEW_PVLIBVIEW_H */
