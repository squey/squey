/**
 * \file PVWidget.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <pvgl/PVWidgetManager.h>
#include <pvgl/PVContainer.h>

#include <pvgl/PVWidget.h>

/******************************************************************************
 *
 * PVGL::PVWidget::PVWidget
 *
 *****************************************************************************/
PVGL::PVWidget::PVWidget(PVWidgetManager *widget_manager_) :
widget_manager(widget_manager_), visible(true)
{
	PVLOG_DEBUG("PVGL::PVWidget::%s\n", __FUNCTION__);

	parent = 0;
	allocation.x = 0;
	allocation.y = 0;
	allocation.width = 10;
	allocation.height= 10;
	requisition.width = 10;
	requisition.height=10;
	min_size.width = -1;
	min_size.height= -1;
}

/******************************************************************************
 *
 * PVGL::PVWidget::~PVWidget
 *
 *****************************************************************************/
PVGL::PVWidget::~PVWidget()
{
	PVLOG_DEBUG("PVGL::PVWidget::%s\n", __FUNCTION__);

}

/******************************************************************************
 *
 * PVGL::PVWidget::move
 *
 *****************************************************************************/
void PVGL::PVWidget::move(int x, int y)
{
	PVLOG_DEBUG("PVGL::PVWidget::%s\n", __FUNCTION__);

	allocation.x = x;
	allocation.y = y;
}

/******************************************************************************
 *
 * PVGL::PVWidget::allocate_size
 *
 *****************************************************************************/
void PVGL::PVWidget::allocate_size(const PVAllocation &new_allocation)
{
	PVLOG_HEAVYDEBUG("PVGL::PVWidget::%s\n", __FUNCTION__);

	allocation = new_allocation;
}

/******************************************************************************
 *
 * PVGL::PVWidget::show
 *
 *****************************************************************************/
void PVGL::PVWidget::show()
{
	PVLOG_DEBUG("PVGL::PVWidget::%s\n", __FUNCTION__);

	visible = true;
	if (parent) {
		parent->size_adjust();
	}
}

/******************************************************************************
 *
 * PVGL::PVWidget::hide
 *
 *****************************************************************************/
void PVGL::PVWidget::hide()
{
	PVLOG_HEAVYDEBUG("PVGL::PVWidget::%s\n", __FUNCTION__);

	visible = false;
	if (parent) {
		parent->size_adjust();
	}
}

/******************************************************************************
 *
 * PVGL::PVWidget::in_allocation
 *
 *****************************************************************************/
bool PVGL::PVWidget::in_allocation(int x, int y)
{
	PVLOG_DEBUG("PVGL::PVWidget::%s\n", __FUNCTION__);

	if (x > allocation.x && x < allocation.x + allocation.width &&
			y > allocation.y && y < allocation.y + allocation.height) {
		return true;
	}
	return false;
}

