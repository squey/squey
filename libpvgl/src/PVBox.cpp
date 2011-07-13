//! \file PVBox.cpp
//! $Id: PVBox.cpp 2520 2011-04-30 12:26:48Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009, 2010
//! Copyright (C) Philippe Saade 2009,2010
//! Copyright (C) Picviz Labs 2011

#include <pvgl/PVWidgetManager.h>

#include <pvgl/PVBox.h>

/******************************************************************************
 *
 * PVGL::PVBox::PVBox
 *
 *****************************************************************************/
PVGL::PVBox::PVBox(PVWidgetManager *widget_manager_) :
PVContainer(widget_manager_)
{
	PVLOG_DEBUG("PVGL::PVBox::%s\n", __FUNCTION__);
}

/******************************************************************************
 *
 * PVGL::PVBox::~PVBox
 *
 *****************************************************************************/
PVGL::PVBox::~PVBox()
{
	PVLOG_DEBUG("PVGL::PVBox::%s\n", __FUNCTION__);

	for (std::list<PVBoxChild>::iterator it = start_list.begin(); it != start_list.end(); ++it) {
		delete it->child;
	}
	for (std::list<PVBoxChild>::iterator it = end_list.begin(); it != end_list.end(); ++it) {
		delete it->child;
	}
}

/******************************************************************************
 *
 * PVGL::PVBox::draw
 *
 *****************************************************************************/
void PVGL::PVBox::draw()
{
	PVLOG_HEAVYDEBUG("PVGL::PVBox::%s\n", __FUNCTION__);

	for (std::list<PVBoxChild>::iterator it = start_list.begin(); it != start_list.end(); ++it) {
		it->child->draw();
	}
	for (std::list<PVBoxChild>::iterator it = end_list.begin(); it != end_list.end(); ++it) {
		it->child->draw();
	}
}

/******************************************************************************
 *
 * PVGL::PVBox::add
 *
 *****************************************************************************/
void PVGL::PVBox::add(PVWidget *child)
{
	PVLOG_DEBUG("PVGL::PVBox::%s\n", __FUNCTION__);

	pack_start(child);
}
