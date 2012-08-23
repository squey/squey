/**
 * \file PVLibView.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVLIBVIEW_H_
#define PVLIBVIEW_H_

#include <picviz/FakePVView.h>

#include <pvhive/PVFuncObserver.h>
#include <pvhive/PVCallHelper.h>

#include <pvparallelview/PVFullParallelScene.h>
#include <pvparallelview/PVZonesManager.h>

namespace PVParallelView
{

class PVZonesManager;
class PVFuncObserverSignal;

class PVLibView
{
private:
	typedef std::list<PVFullParallelScene> views_list_t;
	friend class process_selection_Observer;

public:
	PVLibView(Picviz::FakePVView::shared_pointer& view_sp);
	~PVLibView();

public:
	void create_view(PVParallelView::PVLinesView::zones_drawing_t::bci_backend_t& bci_backend)
	{
		_parallel_views.emplace_back(_view_sp, _zones_manager, bci_backend);
		_parallel_views.back().first_render();
	}

	PVZonesManager& get_zones_manager() { return _zones_manager; }

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
	PVZonesManager _zones_manager;
	process_selection_Observer* _process_selection_observer;
	views_list_t _parallel_views;
	Picviz::FakePVView::shared_pointer& _view_sp;
};

}

#endif /* PVLIBVIEW_H_ */
