//! \file PVIdleManager.cpp
//! $Id: PVIdleManager.cpp 2975 2011-05-26 03:54:42Z aguinet $
//! Copyright (C) SÃ©bastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011
//! Copyright (C) Picviz Labs 2011

#define GLEW_STATIC 1
#include <GL/glew.h>

#include <pvgl/views/PVParallel.h>
#include <pvgl/PVMain.h>
#include <pvgl/PVWTK.h>

#include <pvgl/PVIdleManager.h>

/******************************************************************************
 *
 * PVGL::PVIdleManager::callback
 *
 *****************************************************************************/
void PVGL::PVIdleManager::callback(void)
{
	std::map<IdleTask, IdleValue>::iterator it;

	PVLOG_HEAVYDEBUG("PVGL::PVIdleManager::%s\n", __FUNCTION__);

	for (it = tasks.begin(); it != tasks.end(); ++it) {
		PVDrawable* d = it->first.drawable;
		PVGL::wtk_set_current_window(d->get_window_id());
		d->draw();
		PVGL::wtk_buffers_swap();
	}
	it = tasks.begin();
	while (it != tasks.end()) {
		std::map<IdleTask, IdleValue>::iterator it_next = it;
		it_next++;
		if (it->second.removed) {
			PVLOG_DEBUG("PVGL::PVIdleManager::%s Removing a task.\n", __FUNCTION__);
			tasks.erase(it);
		}
		it = it_next;
	}
	if (tasks.empty()) {
		PVLOG_DEBUG("PVGL::PVIdleManager::%s The Idle List is empty, sleeping.\n", __FUNCTION__);
		PVGL::wtk_set_idle_func(0);
	}
}

/******************************************************************************
 *
 * PVGL::PVIdleManager::new_task
 *
 *****************************************************************************/
void PVGL::PVIdleManager::new_task(PVGL::PVDrawable *drawable, PVGL::PVIdleTaskKinds kind)
{
	IdleTask task(drawable, kind);

	PVLOG_DEBUG("PVGL::PVIdleManager::%s\n", __FUNCTION__);

	tasks[task] = IdleValue(drawable->get_max_lines_per_redraw());
	PVGL::wtk_set_idle_func(PVGL::PVMain::idle_callback);
}

/******************************************************************************
 *
 * PVGL::PVIdleManager::task_exists
 *
 *****************************************************************************/
bool PVGL::PVIdleManager::task_exists(PVGL::PVDrawable *drawable, PVGL::PVIdleTaskKinds kind) const
{
	if (tasks.find(IdleTask(drawable, kind)) != tasks.end()) {
		return true;
	}
	return false;
}

/******************************************************************************
 *
 * PVGL::PVIdleManager::get_number_of_lines
 *
 *****************************************************************************/
int PVGL::PVIdleManager::get_number_of_lines(PVGL::PVDrawable *drawable, PVGL::PVIdleTaskKinds kind)
{
	PVLOG_DEBUG("PVGL::PVIdleManager::%s\n", __FUNCTION__);

	if (drawable->current_scheduler) {
		return (drawable->*(drawable->current_scheduler))(kind);
	}
	return 0;
}

/******************************************************************************
 *
 * PVGL::PVIdleManager::remove_task
 *
 *****************************************************************************/
void PVGL::PVIdleManager::remove_task(PVGL::PVDrawable *drawable, PVGL::PVIdleTaskKinds kind)
{
	IdleTask task(drawable, kind);

	PVLOG_DEBUG("PVGL::PVIdleManager::%s\n", __FUNCTION__);

	if (tasks.find(task) == tasks.end()) {
		return;
	}
	tbb::tick_count end = tbb::tick_count::now();
	PVLOG_INFO("(PVGL::PVIdleManager::remove_task) task done in %0.4f seconds.\n", (end-tasks[task].time_start).seconds());
	IdleValue value(tasks[task].nb_lines, true);
	tasks[task] = value;
}

/******************************************************************************
 *
 * PVGL::PVIdleManager::remove_tasks_for_drawable
 *
 *****************************************************************************/
void PVGL::PVIdleManager::remove_tasks_for_drawable(PVGL::PVDrawable *drawable)
{
	PVLOG_DEBUG("PVGL::PVIdleManager::%s\n", __FUNCTION__);
	std::map<IdleTask, IdleValue>::iterator it;
	
	it = tasks.begin();
	while (it != tasks.end()) {
		std::map<IdleTask, IdleValue>::iterator it_next = it;
		it_next++;
		if ((*it).first.drawable == drawable) {
			PVLOG_DEBUG("PVGL::PVIdleManager::%s Removing a task for a given drawable.\n", __FUNCTION__);
			tasks.erase(it);
		}
		it = it_next;
	}
}
