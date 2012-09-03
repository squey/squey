/**
 * \file PVLinesView.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <tbb/task_group.h>

#include <pvparallelview/PVLibView.h>
#include <pvhive/PVHive.h>

#include <iostream>

PVParallelView::PVLibView::PVLibView(Picviz::FakePVView::shared_pointer& view_sp,
                                     PVCore::PVHSVColor *colors) :
	_process_selection_observer(new process_selection_Observer(this)),
	_view_sp(view_sp),
	_colors(colors)
{
	// Create TBB root task
	_task_root = new (tbb::task::allocate_root(_tasks_ctxt)) tbb::empty_task;
	_task_root->set_ref_count(1);

	PVHive::PVHive::get().register_func_observer(
		view_sp,
		*_process_selection_observer
	);
}

PVParallelView::PVLibView::~PVLibView()
{
	_task_root->destroy(*_task_root);
	delete _process_selection_observer;
}

void PVParallelView::PVLibView::process_selection_Observer::update(arguments_deep_copy_type const& args) const
{
	tbb::task* task_root = _parent->task_root();
	tbb::task_group_context& ctxt = _parent->task_group_context();

	// Cancel current tasks execution
	PVLOG_INFO("PVLibView::process_selection: cancelling current tasks...\n");
	ctxt.cancel_group_execution();
	task_root->wait_for_all();
	PVLOG_INFO("PVLibView::process_selection: tasks stopped.\n");
	ctxt.reset();

	// Flag all zones selection as invalid
	_parent->_zones_manager.invalidate_selection();

	task_root->set_ref_count(1);
	for (PVFullParallelScene& view : _parent->_parallel_views) {
		view.update_new_selection(task_root);
	}
/*
	for (PVZoomedParallelScene& zps : _parent->_zoomed_parallel_scenes) {
		view.update_new_selection(*task_root);
	}*/
}
