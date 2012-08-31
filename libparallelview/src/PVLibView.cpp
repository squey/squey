/**
 * \file PVLinesView.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <tbb/task_group.h>

#include <pvparallelview/PVLibView.h>
#include <pvhive/PVHive.h>

#include <iostream>

void PVParallelView::PVLibView::process_selection_Observer::update(arguments_deep_copy_type const& args) const
{
	// Flag all zones selection as invalid
	_parent->_zones_manager.invalidate_selection();

	tbb::task_group group;
	for (PVFullParallelScene& view : _parent->_parallel_views) {
		group.run([&]{view.update_sel_from_zone();});
	}

	for (PVZoomedParallelScene& zps : _parent->_zoomed_parallel_scenes) {
		group.run([&]
		          {
			          zps.invalidate_selection();
		          });
	}
	group.wait();
}

PVParallelView::PVLibView::PVLibView(Picviz::FakePVView::shared_pointer& view_sp,
                                     PVCore::PVHSVColor *colors) :
	_process_selection_observer(new process_selection_Observer(this)),
	_view_sp(view_sp),
	_colors(colors)
{
	PVHive::PVHive::get().register_func_observer(
		view_sp,
		*_process_selection_observer
	);
}

PVParallelView::PVLibView::~PVLibView()
{
	delete _process_selection_observer;
}
