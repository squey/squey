//! \file PVLabel.cpp
//! $Id: PVLabel.cpp 2520 2011-04-30 12:26:48Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009, 2010
//! Copyright (C) Philippe Saade 2009,2010
//! Copyright (C) Picviz Labs 2011

#define GLEW_STATIC 1
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <pvgl/PVConfig.h>
#include <pvgl/PVUtils.h>
#include <pvgl/PVWidgetManager.h>
#include <pvgl/PVContainer.h>

#include <pvgl/PVLabel.h>

/******************************************************************************
 *
 * PVGL::PVLabel::PVLabel
 *
 *****************************************************************************/
PVGL::PVLabel::PVLabel(PVWidgetManager *widget_manager_, const std::string &text_) :
PVMisc(widget_manager_), text(text_), color(255, 255, 255, 255)
{
	int width, height;
	PVLOG_DEBUG("PVGL::PVLabel::%s\n", __FUNCTION__);

	shadow = false;
	font_size = 14;

	widget_manager->get_text_size(text, font_size, width, height, ascent);
	requisition.width = width + 2 * padding_x;
	requisition.height= height+ 2 * padding_y;

	allocation.width = requisition.width;
	allocation.height= requisition.height;
}

/******************************************************************************
 *
 * PVGL::PVLabel::~PVLabel
 *
 *****************************************************************************/
PVGL::PVLabel::~PVLabel()
{
	PVLOG_DEBUG("PVGL::PVLabel::%s\n", __FUNCTION__);
}

/******************************************************************************
 *
 * PVGL::PVLabel::set_color
 *
 *****************************************************************************/
void PVGL::PVLabel::set_color(const ubvec4 &new_color)
{
	PVLOG_DEBUG("PVGL::PVLabel::%s\n", __FUNCTION__);

	color = new_color;
}

/******************************************************************************
 *
 * PVGL::PVLabel::set_shadow
 *
 *****************************************************************************/
void PVGL::PVLabel::set_shadow(bool do_shadow)
{
	PVLOG_HEAVYDEBUG("PVGL::PVLabel::%s\n", __FUNCTION__);

	shadow = do_shadow;
}

/******************************************************************************
 *
 * PVGL::PVLabel::set_text
 *
 *****************************************************************************/
void PVGL::PVLabel::set_text(const std::string &new_text)
{
	int width, height;

	PVLOG_HEAVYDEBUG("PVGL::PVLabel::%s\n", __FUNCTION__);

	text = new_text;
	widget_manager->get_text_size(text, font_size, width, height, ascent);
	requisition.width = width + 2 * padding_x;
	requisition.height= height+ 2 * padding_y;

	allocation.width = requisition.width;
	allocation.height= requisition.height;
	if (parent)
		parent->size_adjust();
}

/******************************************************************************
 *
 * PVGL::PVLabel::draw
 *
 *****************************************************************************/
void PVGL::PVLabel::draw(void)
{
	float x, y;
	PVLOG_HEAVYDEBUG("PVGL::PVLabel::%s\n", __FUNCTION__);

	if (!visible) {
		return;
	}
	x = allocation.x + (allocation.width - requisition.width) * align_x + padding_x;
	y = allocation.y + (allocation.height- requisition.height)* align_y + padding_y + ascent;
	if (shadow) {
		glColor4ub(0, 0, 0, 128);
		widget_manager->draw_text(x + 1, y + 1, text, font_size);
	}
	glColor4ubv(&color.x);
	widget_manager->draw_text(x, y, text, font_size);
}

/******************************************************************************
 *
 * PVGL::PVLabel::allocate_size
 *
 *****************************************************************************/
void PVGL::PVLabel::allocate_size(const PVAllocation &new_allocation)
{
	PVLOG_HEAVYDEBUG("PVGL::PVLabel::%s\n", __FUNCTION__);

  allocation = new_allocation;
}

