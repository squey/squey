//! \file PVDrawable.cpp
//! $Id$
//! Copyright (C) Sebastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>

#define GLEW_STATIC 1
#include <GL/glew.h>

#include <picviz/PVView.h>

#include <pvsdk/PVMessenger.h>

#include <pvgl/PVConfig.h>
#include <pvgl/PVUtils.h>
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
	PVLOG_HEAVYDEBUG("PVGL::PVDrawable::%s\n", __FUNCTION__);

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
	PVLOG_HEAVYDEBUG("PVGL::PVDrawable::%s\n", __FUNCTION__);

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
PVGL::PVDrawable::PVDrawable(int win_id, PVSDK::PVMessenger *message) :
		pv_message(message),
		widget_manager(0), index(0),
		window_id(win_id),
		frame_count(0)
{
	fps_previous_time = glutGet(GLUT_ELAPSED_TIME);
	// Default width and height needs to be set, or we will use an undefined value
	// when resizing our layouts
	width = pvconfig.value("pvgl/parallel_view_width", PVGL_VIEW_DEFAULT_WIDTH).toInt();
	height = pvconfig.value("pvgl/parallel_view_height", PVGL_VIEW_DEFAULT_HEIGHT).toInt();

	PVLOG_HEAVYDEBUG("PVGL::PVDrawable::%s\n", __FUNCTION__);
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
	PVLOG_HEAVYDEBUG("PVGL::PVDrawable::%s\n", __FUNCTION__);
	picviz_view = view;

	int max_lines_for_scheduler_small = pvconfig.value("pvgl/max_lines_for_scheduler_small", 80000).toInt();

	if (picviz_view->get_row_count() < max_lines_for_scheduler_small) {
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
void PVGL::PVDrawable::compute_fps()
{
	frame_count++;
	
	int cur_time = glutGet(GLUT_ELAPSED_TIME);
	int time_interval = cur_time - fps_previous_time;

	if(time_interval > 200)
	{
		// calculate the number of frames per second
		current_fps = (double) frame_count / ((double) time_interval / 1000.0);

		// Set time
		fps_previous_time = cur_time;

		// Reset frame count
		frame_count = 0;
	}
}

/******************************************************************************
 *
 * PVGL::PVDrawable::set_size
 *
 *****************************************************************************/
void PVGL::PVDrawable::set_size(int w, int h)
{
	PVLOG_HEAVYDEBUG("PVGL::PVDrawable::%s\n", __FUNCTION__);

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
	PVLOG_HEAVYDEBUG("PVGL::PVDrawable::%s\n", __FUNCTION__);
}

/******************************************************************************
 *
 * PVGL::PVDrawable::special_keys
 *
 *****************************************************************************/
void PVGL::PVDrawable::special_keys(int, int, int)
{
	PVLOG_HEAVYDEBUG("PVGL::PVDrawable::%s\n", __FUNCTION__);
}

/******************************************************************************
 *
 * PVGL::PVDrawable::mouse_wheel
 *
 *****************************************************************************/
void PVGL::PVDrawable::mouse_wheel(int, int, int)
{
	PVLOG_HEAVYDEBUG("PVGL::PVDrawable::%s\n", __FUNCTION__);
}

/******************************************************************************
 *
 * PVGL::PVDrawable::mouse_down
 *
 *****************************************************************************/
void PVGL::PVDrawable::mouse_down(int, int, int, int)
{
	PVLOG_HEAVYDEBUG("PVGL::PVDrawable::%s\n", __FUNCTION__);
}

/******************************************************************************
 *
 * PVGL::PVDrawable::mouse_move
 *
 *****************************************************************************/
bool PVGL::PVDrawable::mouse_move(int, int, int)
{
	PVLOG_HEAVYDEBUG("PVGL::PVDrawable::%s\n", __FUNCTION__);

	return false;
}

/******************************************************************************
 *
 * PVGL::PVDrawable::mouse_up
 *
 *****************************************************************************/
bool PVGL::PVDrawable::mouse_up(int, int, int, int)
{
	PVLOG_HEAVYDEBUG("PVGL::PVDrawable::%s\n", __FUNCTION__);

	return false;
}

/******************************************************************************
 *
 * PVGL::PVDrawable::passive_motion
 *
 *****************************************************************************/
bool PVGL::PVDrawable::passive_motion(int, int, int)
{
	PVLOG_HEAVYDEBUG("PVGL::PVDrawable::%s\n", __FUNCTION__);

	return false;
}

/******************************************************************************
 *
 * PVGL::PVDrawable::get_max_lines_per_redraw
 *
 *****************************************************************************/
int PVGL::PVDrawable::get_max_lines_per_redraw() const
{
	PVLOG_HEAVYDEBUG("PVGL::PVDrawable::%s\n", __FUNCTION__);

	return max_lines_per_redraw;
}

/******************************************************************************
 *
 * PVGL::PVDrawable::update_views
 *
 *****************************************************************************/
void PVGL::PVDrawable::update_views() const
{
	QList<Picviz::PVView*> views_to_update;
	picviz_view->emit_user_modified_sel(&views_to_update);
	PVMain::update_views_sel(views_to_update);
}
