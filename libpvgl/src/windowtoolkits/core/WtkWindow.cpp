//! \file WtkWindow.cpp
//! Functions that hide the window to the pvgl code
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011

#include <picviz/PVView.h>

#include <pvgl/PVConfig.h>
#include <pvgl/PVUtils.h>
// #include <pvgl/PVIdleManager.h>
#include <pvgl/PVMain.h>

#include "include/WtkWindow.h"

void PVGL::WTK::WtkWindow::_common_init(void)
{
	max_lines_per_redraw = 1;
}

PVGL::WTK::WtkWindow::WtkWindow(PVSDK::PVMessenger *com, int win_id) 
{
	_win_id = win_id;
	_win_type = WTK_WINDOWTYPE_INT;

	_common_init();
}

PVGL::WTK::WtkWindow::WtkWindow(PVSDK::PVMessenger *com, void *win_ptr) 
{
	_win_ptr = win_ptr;
	_win_type = WTK_WINDOWTYPE_POINTER;

	_common_init();
}

PVGL::WTK::WtkWindow::~WtkWindow() {}

void PVGL::WTK::WtkWindow::init(Picviz::PVView_p view)
{
	PVLOG_DEBUG("PVGL::WTK::WtkWindow::%s\n", __FUNCTION__);
	picviz_view = view;

	PVLOG_INFO("We are in the init() function\n");

	int max_lines_for_scheduler_small = pvconfig.value("pvgl/max_lines_for_scheduler_small", 80000).toInt();

	// if (picviz_view->get_row_count() < max_lines_for_scheduler_small) {
	// 	current_scheduler = &PVGL::WTK::WtkWindow::small_files_scheduler;
	// } else {
	// 	current_scheduler = &PVGL::WTK::WtkWindow::dumb_scheduler;
	// }
}

// /******************************************************************************
//  *
//  * PVGL::WTK::WtkWindow::dumb_scheduler
//  *
//  *****************************************************************************/
// int PVGL::WTK::WtkWindow::dumb_scheduler(PVGL::PVIdleTaskKinds kind)
// {
// 	PVLOG_DEBUG("PVGL::WTK::WtkWindow::%s\n", __FUNCTION__);

// 	if (!idle_manager.task_exists(this, kind)) {
// 		return 0;
// 	}

// 	switch (kind) {
// 		case IDLE_REDRAW_ZOMBIE_LINES: // Don't draw zombie lines if there are selected lines waiting.
// 				if (idle_manager.task_exists(this, IDLE_REDRAW_LINES)) {
// 					return get_max_lines_per_redraw() / 10;
// 				}
// 				break;
// 		case IDLE_REDRAW_MAP_LINES:
// 				if (idle_manager.task_exists(this, IDLE_REDRAW_LINES) ||
// 					  idle_manager.task_exists(this, IDLE_REDRAW_ZOMBIE_LINES)) {
// 					return get_max_lines_per_redraw() / 100;
// 				}
// 				break;
// 		case IDLE_REDRAW_ZOMBIE_MAP_LINES:
// 				if (idle_manager.task_exists(this, IDLE_REDRAW_LINES) ||
// 						idle_manager.task_exists(this, IDLE_REDRAW_ZOMBIE_LINES)) {
// 					return get_max_lines_per_redraw() / 100;
// 				}
// 				if (idle_manager.task_exists(this, IDLE_REDRAW_MAP_LINES)) {
// 					return get_max_lines_per_redraw() / 10;
// 				}
// 				break;
// 		default:
// 				return get_max_lines_per_redraw();
// 	}
// 	return get_max_lines_per_redraw();
// }

// /******************************************************************************
//  *
//  * PVGL::WTK::WtkWindow::small_files_scheduler
//  *
//  *****************************************************************************/
// int PVGL::WTK::WtkWindow::small_files_scheduler(PVGL::PVIdleTaskKinds kind)
// {
// 	PVLOG_DEBUG("PVGL::WTK::WtkWindow::%s\n", __FUNCTION__);

// 	if (!idle_manager.task_exists(this, kind)) {
// 		return 0;
// 	}

// 	return picviz_view->get_row_count() + 1;
// }



/******************************************************************************
 *
 * PVGL::WTK::WtkWindow::set_size
 *
 *****************************************************************************/
void PVGL::WTK::WtkWindow::set_size(int w, int h)
{
	PVLOG_DEBUG("PVGL::WTK::WtkWindow::%s\n", __FUNCTION__);

	width = w;
	height = h;
	glViewport(0, 0, width, height);
}

/******************************************************************************
 *
 * PVGL::WTK::WtkWindow::keyboard
 *
 *****************************************************************************/
void PVGL::WTK::WtkWindow::keyboard(unsigned char, int, int)
{
	PVLOG_DEBUG("PVGL::WTK::WtkWindow::%s\n", __FUNCTION__);
}

/******************************************************************************
 *
 * PVGL::WTK::WtkWindow::special_keys
 *
 *****************************************************************************/
void PVGL::WTK::WtkWindow::special_keys(int, int, int)
{
	PVLOG_DEBUG("PVGL::WTK::WtkWindow::%s\n", __FUNCTION__);
}

/******************************************************************************
 *
 * PVGL::WTK::WtkWindow::mouse_wheel
 *
 *****************************************************************************/
void PVGL::WTK::WtkWindow::mouse_wheel(int, int, int)
{
	PVLOG_DEBUG("PVGL::WTK::WtkWindow::%s\n", __FUNCTION__);
}

/******************************************************************************
 *
 * PVGL::WTK::WtkWindow::mouse_down
 *
 *****************************************************************************/
void PVGL::WTK::WtkWindow::mouse_down(int, int, int, int)
{
	PVLOG_DEBUG("PVGL::WTK::WtkWindow::%s\n", __FUNCTION__);
}

/******************************************************************************
 *
 * PVGL::WTK::WtkWindow::mouse_move
 *
 *****************************************************************************/
bool PVGL::WTK::WtkWindow::mouse_move(int, int, int)
{
	PVLOG_DEBUG("PVGL::WTK::WtkWindow::%s\n", __FUNCTION__);

	return false;
}

/******************************************************************************
 *
 * PVGL::WTK::WtkWindow::mouse_up
 *
 *****************************************************************************/
bool PVGL::WTK::WtkWindow::mouse_up(int, int, int, int)
{
	PVLOG_DEBUG("PVGL::WTK::WtkWindow::%s\n", __FUNCTION__);

	return false;
}

/******************************************************************************
 *
 * PVGL::WTK::WtkWindow::passive_motion
 *
 *****************************************************************************/
bool PVGL::WTK::WtkWindow::passive_motion(int, int, int)
{
	PVLOG_DEBUG("PVGL::WTK::WtkWindow::%s\n", __FUNCTION__);

	return false;
}

