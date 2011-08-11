//! \file PVLayout.cpp
//! $Id: PVLayout.cpp 2520 2011-04-30 12:26:48Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <pvgl/PVWidgetManager.h>

#include <pvgl/PVLayout.h>

/******************************************************************************
 *
 * PVGL::PVLayout::PVLayout
 *
 *****************************************************************************/
PVGL::PVLayout::PVLayout(PVWidgetManager *widget_manager_) : PVContainer(widget_manager_)
{
	PVLOG_DEBUG("PVGL::PVLayout::%s\n", __FUNCTION__);

}

/******************************************************************************
 *
 * PVGL::PVLayout::draw
 *
 *****************************************************************************/
void PVGL::PVLayout::draw()
{
	PVLOG_HEAVYDEBUG("PVGL::PVLayout::%s\n", __FUNCTION__);

	if (!visible)
		return;

	// Draw the layout container itself.
	glEnable(GL_BLEND);
	glColor4f(0.6f, 0.7f, 0.9f, 0.8f);
	glBegin(GL_QUADS);
	glVertex2f(allocation.x,                    allocation.y);
	glVertex2f(allocation.x + allocation.width, allocation.y);
	glVertex2f(allocation.x + allocation.width, allocation.y + allocation.height);
	glVertex2f(allocation.x,                    allocation.y + allocation.height);
	glEnd();

	// Then draw the children
	for (std::list<LayoutChild>::iterator it = children.begin(); it != children.end(); ++it) {
		it->widget->draw();
	}
}

/******************************************************************************
 *
 * PVGL::PVLayout::add
 *
 *****************************************************************************/
void PVGL::PVLayout::add(PVWidget *child, int x1, int y1, int x2, int y2)
{
	PVLOG_DEBUG("PVGL::PVLayout::%s\n", __FUNCTION__);

	LayoutChild new_child;
	new_child.widget = child;
	new_child.x1 = x1;
	new_child.y1 = y1;
	new_child.x2 = x2;
	new_child.y2 = y2;
	children.push_back(new_child);
	child->set_parent(this);
	size_adjust();
}

/******************************************************************************
 *
 * PVGL::PVLayout::add
 *
 *****************************************************************************/
void PVGL::PVLayout::add(PVWidget *child)
{
	PVLOG_DEBUG("PVGL::PVLayout::%s\n", __FUNCTION__);

	add(child, 0, 0, -1, -1);
}

/******************************************************************************
 *
 * PVGL::PVLayout::set_size
 *
 *****************************************************************************/
void PVGL::PVLayout::set_size(int width, int height)
{
	PVLOG_DEBUG("PVGL::PVLayout::%s\n", __FUNCTION__);

	requisition.width = width;
	requisition.height= height;

	size_adjust();
}

/******************************************************************************
 *
 * PVGL::PVLayout::allocate_size
 *
 *****************************************************************************/
void PVGL::PVLayout::allocate_size(const PVAllocation &new_allocation)
{
	PVLOG_HEAVYDEBUG("PVGL::PVLayout::%s\n", __FUNCTION__);

	allocation = new_allocation;

	for (std::list<LayoutChild>::iterator it = children.begin(); it != children.end(); ++it) {
		PVAllocation child_allocation;
		if (it->x1 >= 0) {
			child_allocation.x = allocation.x + it->x1;
		} else {
			child_allocation.x = allocation.x + allocation.width + it->x1;
		}
		if (it->y1 >= 0) {
			child_allocation.y = allocation.y + it->y1;
		} else {
			child_allocation.y = allocation.y + allocation.height + it->y1;
		}
		if (it->x2 >= 0) {
			child_allocation.width = it->x2 - child_allocation.x;
		} else {
			child_allocation.width = allocation.x + allocation.width + it->x2 - child_allocation.x;
		}
		child_allocation.width = picviz_max(1, child_allocation.width);
		if (it->y2 >= 0) {
			child_allocation.height = it->y2 - child_allocation.y;
		} else {
			child_allocation.height = allocation.y + allocation.height + it->y2 - child_allocation.y;
		}
		child_allocation.height = picviz_max(1, child_allocation.height);
		it->widget->allocate_size(child_allocation);
	}
}

/******************************************************************************
 *
 * PVGL::PVLayout::size_adjust
 *
 *****************************************************************************/
void PVGL::PVLayout::size_adjust()
{
	PVLOG_HEAVYDEBUG("PVGL::PVLayout::%s\n", __FUNCTION__);

	if (parent) {
		parent->size_adjust();
	} else {
		PVAllocation new_allocation;
		new_allocation.x = allocation.x;
		new_allocation.y = allocation.y;
		new_allocation.width = requisition.width;
		new_allocation.height= requisition.height;
		allocate_size(new_allocation);
	}
}
