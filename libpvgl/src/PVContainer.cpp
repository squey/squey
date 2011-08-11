//! \file PVContainer.cpp
//! $Id: PVContainer.cpp 2520 2011-04-30 12:26:48Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <pvgl/PVWidgetManager.h>

#include <pvgl/PVContainer.h>

/******************************************************************************
 *
 * PVGL::PVContainer::PVGL::PVContainer
 *
 *****************************************************************************/
PVGL::PVContainer::PVContainer(PVWidgetManager *widget_manager_) :
PVWidget(widget_manager_)
{
	PVLOG_DEBUG("PVGL::PVContainer::%s\n", __FUNCTION__);

	border_width = 2;
}

/******************************************************************************
 *
 * PVGL::PVContainer::set_border_width
 *
 *****************************************************************************/
void PVGL::PVContainer::set_border_width(int new_border_width)
{
	PVLOG_DEBUG("PVGL::PVContainer::%s\n", __FUNCTION__);

	border_width = new_border_width;
	size_adjust();
}
