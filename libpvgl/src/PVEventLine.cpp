//! \file PVEventLine.cpp
//! $Id: PVEventLine.cpp 2875 2011-05-19 04:18:05Z aguinet $
//! Copyright (C) SÃ©bastien Tricaud 2009-2011
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

#include <pvgl/PVConfig.h>
#include <pvgl/PVUtils.h>
#include <pvgl/PVCom.h>
#include <pvgl/views/PVParallel.h>

#include <pvgl/PVEventLine.h>

#include <pvgl/PVWTK.h>

/******************************************************************************
 *
 * PVGL::PVEventLine::PVEventLine
 *
 *****************************************************************************/
PVGL::PVEventLine::PVEventLine(PVWidgetManager *widget_manager, PVView *pvgl_view, PVCom *com) :
PVWidget(widget_manager), view(pvgl_view), pv_com(com)
{
	PVLOG_DEBUG("PVGL::PVEventLine::%s\n", __FUNCTION__);

	last_mouse_press_position_x = last_mouse_press_position_y = -1;
	sliders_positions[0] = 0.0;
	sliders_positions[1] = 1.0;
	sliders_positions[2] = 1.0;
	prelight[0] = prelight[1] = prelight[2] = false;

	grabbed_slider = -1;
	grabbing = false;

	max_lines_interactivity = pvconfig.value("pvgl/max_lines_for_interactivity", MAX_LINES_FOR_INTERACTIVITY).toInt();

}

/******************************************************************************
 *
 * PVGL::PVEventLine::set_view
 *
 *****************************************************************************/
void PVGL::PVEventLine::set_view(Picviz::PVView_p picviz_view_)
{
	PVLOG_DEBUG("PVGL::PVEventLine::%s\n", __FUNCTION__);

	picviz_view = picviz_view_;
}

/******************************************************************************
 *
 * PVGL::PVEventLine::set_requisition
 *
 *****************************************************************************/
void PVGL::PVEventLine::set_size(int width, int height)
{
	PVLOG_DEBUG("PVGL::PVEventLine::%s\n", __FUNCTION__);

	requisition.width = width;
	requisition.height=height;

	if (parent) {
		parent->size_adjust();
	}
}

/******************************************************************************
 *
 * PVGL::PVEventLine::draw
 *
 *****************************************************************************/
void PVGL::PVEventLine::draw(void)
{
	vec2 pos;

	PVLOG_HEAVYDEBUG("PVGL::PVEventLine::%s\n", __FUNCTION__);

	glDisable(GL_DEPTH_TEST);
	for (unsigned i = 0; i < 3; i++) {
		sliders_positions[i] = picviz_view->eventline.get_kth_slider_position(i);
	}
	// The background
	widget_manager->draw_icon_streched(allocation.x, allocation.y, allocation.width, allocation.height, PVGL_ICON_EVENTLINE_BACK);
	// The central line
	widget_manager->draw_icon_streched(allocation.x + 20, allocation.y, allocation.width - 40, allocation.height, PVGL_ICON_EVENTLINE_LINE);

	pos.y = allocation.y + allocation.height / 2;
	// The left slider
	pos.x = int(allocation.x + 10 + 10 + (allocation.width - 40) * sliders_positions[0]);
	if (prelight[0]) {
		widget_manager->draw_icon(pos.x, pos.y, PVGL_ICON_EVENTLINE_LEFT_SLIDER_PRELIGHT);
	} else {
		widget_manager->draw_icon(pos.x, pos.y, PVGL_ICON_EVENTLINE_LEFT_SLIDER);
	}

	// The right slider
	pos.x = int(allocation.x + 10 + 10 + (allocation.width - 40) * sliders_positions[2]);
	if (prelight[2]) {
		widget_manager->draw_icon(pos.x, pos.y, PVGL_ICON_EVENTLINE_RIGHT_SLIDER_PRELIGHT);
	} else {
		widget_manager->draw_icon(pos.x, pos.y, PVGL_ICON_EVENTLINE_RIGHT_SLIDER);
	}

	// The middle slider
	pos.x = int(allocation.x + 10 + 10 + (allocation.width - 40) * sliders_positions[1]);
	if (prelight[1]) {
		widget_manager->draw_icon(pos.x, pos.y, PVGL_ICON_EVENTLINE_THUMB_PRELIGHT);
	} else {
		widget_manager->draw_icon(pos.x, pos.y, PVGL_ICON_EVENTLINE_THUMB);
	}
}

/******************************************************************************
 *
 * PVGL::PVEventLine::mouse_down
 *
 *****************************************************************************/
bool PVGL::PVEventLine::mouse_down(int /*button*/, int x, int y, int /*modifiers*/)
{
	PVLOG_DEBUG("PVGL::PVEventLine::%s\n", __FUNCTION__);

	if (!visible) {
		return false;
	}
	if (x > allocation.x && x < allocation.x + allocation.width &&
			y > allocation.y && y < allocation.y + allocation.height) {
		float pos_x[3];
		for (unsigned i = 0 ; i < 3; i++) {
			pos_x[i] = allocation.x + 10 + 10 + (allocation.width - 40) * sliders_positions[i];
		}
		grabbing = true;
		// chek if we are close to a slider
		if (x > pos_x[1] - 10 && x < pos_x[1] + 10) {
			grabbed_slider = 1;
			mouse_diff = x - pos_x[1];
		} else if (x > pos_x[0] - 17 && x < pos_x[0]) {
			mouse_diff = x - pos_x[0];
			grabbed_slider = 0;
		} else if (x > pos_x[2] && x < pos_x[2] + 17) {
			mouse_diff = x - pos_x[2];
			grabbed_slider = 2;
		} else {
			grabbed_slider = -1;
		}
		return true;
	}
	return false;
}

/******************************************************************************
 *
 * PVGL::PVEventLine::mouse_move
 *
 *****************************************************************************/
bool PVGL::PVEventLine::mouse_move(int x, int /*y*/, int /*modifiers*/)
{
	PVLOG_DEBUG("PVGL::PVEventLine::%s\n", __FUNCTION__);

	if (!visible) {
		return false;
	}
	if (grabbing) {
		if (grabbed_slider != -1) {
			float pos_x = (x - mouse_diff - (allocation.x + 10 + 10)) / float (allocation.width - 40);
			switch (grabbed_slider) {
				case 0:
						pos_x = std::max(0.0f, std::min(sliders_positions[1], pos_x));
						break;
				case 1:
						pos_x = std::max(sliders_positions[0], std::min(sliders_positions[2], pos_x));
						break;
				case 2:
						pos_x = std::max(sliders_positions[1], std::min(1.0f, pos_x));
						break;
			}
			//sliders_positions[grabbed_slider] = picviz_eventline_set_kth_index_and_adjust_slider_position (picviz_view->eventline, grabbed_slider, pos_x);
			sliders_positions[grabbed_slider] = picviz_view->eventline.set_kth_index_and_adjust_slider_position(grabbed_slider, pos_x);
			PVGL::wtk_window_need_redisplay();
			if (picviz_view->eventline.get_row_count() < max_lines_interactivity) {
				view->get_lines().update_arrays_selection();
			}
		}
		return true;
	}
	return false;
}

/******************************************************************************
 *
 * PVGL::PVEventLine::mouse_up
 *
 *****************************************************************************/
bool PVGL::PVEventLine::mouse_up(int /*button*/, int /*x*/, int /*y*/, int /*modifiers*/)
{
	PVLOG_DEBUG("PVGL::PVEventLine::%s\n", __FUNCTION__);

	if (!visible) {
		return false;
	}
	if (grabbing) {
		grabbing = false;
		if (grabbed_slider != -1) {
			PVGL::PVMessage message;

			message.function = PVGL_COM_FUNCTION_REFRESH_LISTING;
			message.pv_view = picviz_view;
			pv_com->post_message_to_qt(message);
			message.function = PVGL_COM_FUNCTION_SELECTION_CHANGED;
			message.function = PVGL_COM_FUNCTION_REFRESH_LISTING;
			
			if (picviz_view->eventline.get_row_count() >= max_lines_interactivity) {
				view->get_lines().update_arrays_selection();
			}
			PVGL::wtk_window_need_redisplay();
		}

		return true;
	}
	return false;
}

/******************************************************************************
 *
 * PVGL::PVEventLine::passive_motion
 *
 *****************************************************************************/
bool PVGL::PVEventLine::passive_motion(int x, int y, int /*modifiers*/)
{
	PVLOG_HEAVYDEBUG("PVGL::PVEventLine::%s\n", __FUNCTION__);

	if (!visible) {
		return false;
	}
	if (x > allocation.x && x < allocation.x + allocation.width &&
			y > allocation.y && y < allocation.y + allocation.height) {
		float pos_x[3];
		for (unsigned i = 0 ; i < 3; i++) {
			pos_x[i] = allocation.x + 10 + 10 + (allocation.width - 40) * sliders_positions[i];
		}
		// chek if we are close to a slider
		prelight[0] = prelight[1] = prelight[2] = false;
		if (x > pos_x[1] - 10 && x < pos_x[1] + 10) {
			prelight[1] = true;
		} else if (x > pos_x[0] - 17 && x < pos_x[0]) {
			prelight[0] = true;
		} else if (x > pos_x[2] && x < pos_x[2] + 17) {
			prelight[2] = true;
		}
		PVGL::wtk_window_need_redisplay();
		return true;
	}

	return false;
}
