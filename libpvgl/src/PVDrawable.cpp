//! \file PVDrawable.cpp
//! $Id: PVDrawable.cpp 2985 2011-05-26 09:01:11Z dindinx $
//! Copyright (C) SÃÂ©bastien Tricaud 2009, 2010
//! Copyright (C) Philippe Saade 2009,2010
//! Copyright (C) Picviz Labs 2011

#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>

#define GLEW_STATIC 1
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <picviz/PVView.h>

#include <pvgl/PVConfig.h>
#include <pvgl/PVUtils.h>
#include <pvgl/PVCom.h>
#include <pvgl/PVIdleManager.h>
#include <pvgl/PVMain.h>

#include <pvgl/PVDrawable.h>

/******************************************************************************
 *
 * PVGL::PVDrawable::dumb_scheduler
 *
 *****************************************************************************/
int PVGL::PVDrawable::dumb_scheduler(PVGL::PVIdleTaskKinds kind)
{
	PVLOG_DEBUG("PVGL::PVDrawable::%s\n", __FUNCTION__);

	if (!idle_manager.task_exists(this, kind)) {
		return 0;
	}

	switch (kind) {
		case IDLE_REDRAW_ZOMBIE_LINES: // Don't draw zombie lines if there are selected lines waiting.
				if (idle_manager.task_exists(this, IDLE_REDRAW_LINES)) {
					return get_max_lines_per_redraw() / 10;
				}
				break;
		case IDLE_REDRAW_MAP_LINES:
				if (idle_manager.task_exists(this, IDLE_REDRAW_LINES) ||
					  idle_manager.task_exists(this, IDLE_REDRAW_ZOMBIE_LINES)) {
					return get_max_lines_per_redraw() / 100;
				}
				break;
		case IDLE_REDRAW_ZOMBIE_MAP_LINES:
				if (idle_manager.task_exists(this, IDLE_REDRAW_LINES) ||
						idle_manager.task_exists(this, IDLE_REDRAW_ZOMBIE_LINES)) {
					return get_max_lines_per_redraw() / 100;
				}
				if (idle_manager.task_exists(this, IDLE_REDRAW_MAP_LINES)) {
					return get_max_lines_per_redraw() / 10;
				}
				break;
		default:
				return get_max_lines_per_redraw();
	}
	return get_max_lines_per_redraw();
}

/******************************************************************************
 *
 * PVGL::PVDrawable::small_files_scheduler
 *
 *****************************************************************************/
int PVGL::PVDrawable::small_files_scheduler(PVGL::PVIdleTaskKinds kind)
{
	PVLOG_DEBUG("PVGL::PVDrawable::%s\n", __FUNCTION__);

	if (!idle_manager.task_exists(this, kind)) {
		return 0;
	}

	return picviz_view->get_row_count() + 1;
}

/******************************************************************************
 *
 * PVGL::PVDrawable::PVDrawable
 *
 *****************************************************************************/
PVGL::PVDrawable::PVDrawable(int win_id, PVCom *com) :
		pv_com(com),
		width(PVGL_VIEW_DEFAULT_WIDTH), height(PVGL_VIEW_DEFAULT_HEIGHT),
		widget_manager(0), index(0),
		window_id(win_id)
{
	PVLOG_DEBUG("PVGL::PVDrawable::%s\n", __FUNCTION__);
	fullscreen = false;
}

PVGL::PVDrawable::~PVDrawable()
{
}

/******************************************************************************
 *
 * PVGL::PVDrawable::init
 *
 *****************************************************************************/
void PVGL::PVDrawable::init(Picviz::PVView_p view)
{
	PVLOG_DEBUG("PVGL::PVDrawable::%s\n", __FUNCTION__);
	picviz_view = view;

	if (picviz_view->get_row_count() < 80000) {
		current_scheduler = &PVGL::PVDrawable::small_files_scheduler;
	} else {
		current_scheduler = &PVGL::PVDrawable::dumb_scheduler;
	}
}

/******************************************************************************
 *
 * PVGL::PVDrawable::set_size
 *
 *****************************************************************************/
void PVGL::PVDrawable::set_size(int w, int h)
{
	PVLOG_DEBUG("PVGL::PVDrawable::%s\n", __FUNCTION__);

	width = w;
	height = h;
	glViewport(0, 0, width, height);
}

/******************************************************************************
 *
 * PVGL::PVDrawable::keyboard
 *
 *****************************************************************************/
void PVGL::PVDrawable::keyboard(unsigned char, int, int)
{
	PVLOG_DEBUG("PVGL::PVDrawable::%s\n", __FUNCTION__);
}

/******************************************************************************
 *
 * PVGL::PVDrawable::special_keys
 *
 *****************************************************************************/
void PVGL::PVDrawable::special_keys(int, int, int)
{
	PVLOG_DEBUG("PVGL::PVDrawable::%s\n", __FUNCTION__);
}

/******************************************************************************
 *
 * PVGL::PVDrawable::mouse_wheel
 *
 *****************************************************************************/
void PVGL::PVDrawable::mouse_wheel(int, int, int)
{
	PVLOG_DEBUG("PVGL::PVDrawable::%s\n", __FUNCTION__);
}

/******************************************************************************
 *
 * PVGL::PVDrawable::mouse_down
 *
 *****************************************************************************/
void PVGL::PVDrawable::mouse_down(int, int, int, int)
{
	PVLOG_DEBUG("PVGL::PVDrawable::%s\n", __FUNCTION__);
}

/******************************************************************************
 *
 * PVGL::PVDrawable::mouse_move
 *
 *****************************************************************************/
bool PVGL::PVDrawable::mouse_move(int, int, int)
{
	PVLOG_DEBUG("PVGL::PVDrawable::%s\n", __FUNCTION__);

	return false;
}

/******************************************************************************
 *
 * PVGL::PVDrawable::mouse_up
 *
 *****************************************************************************/
bool PVGL::PVDrawable::mouse_up(int, int, int, int)
{
	PVLOG_DEBUG("PVGL::PVDrawable::%s\n", __FUNCTION__);

	return false;
}

/******************************************************************************
 *
 * PVGL::PVDrawable::passive_motion
 *
 *****************************************************************************/
bool PVGL::PVDrawable::passive_motion(int, int, int)
{
	PVLOG_DEBUG("PVGL::PVDrawable::%s\n", __FUNCTION__);

	return false;
}

/******************************************************************************
 *
 * PVGL::PVDrawable::get_max_lines_per_redraw
 *
 *****************************************************************************/
int PVGL::PVDrawable::get_max_lines_per_redraw() const
{
	return max_lines_per_redraw;
}
