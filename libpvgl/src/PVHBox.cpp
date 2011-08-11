//! \file PVHBox.cpp
//! $Id: PVHBox.cpp 2520 2011-04-30 12:26:48Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <pvgl/PVWidgetManager.h>

#include <pvgl/PVHBox.h>

/******************************************************************************
 *
 * PVGL::PVHBox::PVHBox
 *
 *****************************************************************************/
PVGL::PVHBox::PVHBox(PVWidgetManager *widget_manager_) :
PVBox(widget_manager_)
{
	PVLOG_DEBUG("PVGLHBox::%s\n", __FUNCTION__);

}

/******************************************************************************
 *
 * PVGL::PVHBox::allocate_size
 *
 *****************************************************************************/
void PVGL::PVHBox::allocate_size(const PVAllocation &new_allocation)
{
	int requisition_width = 0;
	int min_requisition_width = 0;
	int children_number = 0;
	int expandable_number = 0;
	int shrinkable_children = 0;
	int extra_amount;
	int min_extra_amount;
  PVAllocation child_allocation;

	PVLOG_HEAVYDEBUG("PVGLHBox::%s\n", __FUNCTION__);

	allocation = new_allocation;
	for (std::list<PVBoxChild>::iterator it = start_list.begin(); it != start_list.end(); ++it) {
		if (it->child->is_visible()) {
			requisition_width += it->child->get_requisition().width;
			if (it->child->min_size.width != -1) {
				shrinkable_children++;
				min_requisition_width += it->child->min_size.width;
			} else {
				min_requisition_width += it->child->get_requisition().width;
			}
			children_number++;
			if (it->expand) {
				expandable_number++;
			}
		}
	}
	if (children_number == 0) {
		return;
	}
	extra_amount = new_allocation.width - requisition.width - 2 * border_width;
	min_extra_amount = new_allocation.width - min_requisition_width - 2 * border_width;

	child_allocation.x = new_allocation.x + border_width;
	child_allocation.y = new_allocation.y + border_width;
	child_allocation.height = new_allocation.height - 2 * border_width;

	if (extra_amount < 0) {
		for (std::list<PVBoxChild>::iterator it = start_list.begin(); it != start_list.end(); ++it) {
			if (it->child->is_visible()) {
				if (it->child->min_size.width != -1) {
					int extra = min_extra_amount / shrinkable_children;
					child_allocation.width = it->child->min_size.width + extra;
					min_extra_amount -= extra;
					shrinkable_children--;
				} else {
					child_allocation.width = it->child->get_requisition().width;
				}
				it->child->allocate_size(child_allocation);
				child_allocation.x += child_allocation.width;
			}
		}
	} else {
		for (std::list<PVBoxChild>::iterator it = start_list.begin(); it != start_list.end(); ++it) {
			int extra;

			if (expandable_number) {
				extra = extra_amount / expandable_number;
			} else {
				extra = 0;
			}

			if (it->child->is_visible()) {
				if (it->expand) {
					child_allocation.width = it->child->get_requisition().width + extra;
					it->child->allocate_size(child_allocation);
					child_allocation.x += child_allocation.width;
					extra_amount -= extra;
					expandable_number--;
				} else {
					child_allocation.width = it->child->get_requisition().width;
					it->child->allocate_size(child_allocation);
					child_allocation.x += child_allocation.width;
					expandable_number--;
				}
			}
		}
	}
}

/******************************************************************************
 *
 * PVGL::PVHBox::size_adjust
 *
 *****************************************************************************/
void PVGL::PVHBox::size_adjust()
{
	PVLOG_HEAVYDEBUG("PVGLHBox::%s\n", __FUNCTION__);

  requisition.width = 2 * border_width;
	requisition.height= 2 * border_width;

	for (std::list<PVBoxChild>::iterator it = start_list.begin(); it != start_list.end(); ++it) {
		if (it->child->is_visible()) {
			requisition.width += it->child->get_requisition().width;
			requisition.height = std::max(requisition.height, 2 * border_width + it->child->get_requisition().height);
		}
	}
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

/******************************************************************************
 *
 * PVGL::PVHBox::pack_start
 *
 *****************************************************************************/
void PVGL::PVHBox::pack_start(PVWidget *child, bool expand)
{
	PVLOG_DEBUG("PVGL::PVHBox::%s\n", __FUNCTION__);

	if (child->is_visible()) {
		if (!start_list.empty()) {
			requisition.width += child->get_requisition().width;
			requisition.height = std::max(requisition.height, 2 * border_width + child->get_requisition().height);
		} else {
			requisition.width = 2 *border_width + child->get_requisition().width;
			requisition.height= 2 *border_width + child->get_requisition().height;
		}
	}
	PVBoxChild box_child(child, expand);

	start_list.push_back(PVBoxChild(child, expand));
	child->set_parent(this);
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

/******************************************************************************
 *
 * PVGL::PVHBox::pack_end
 *
 *****************************************************************************/
void PVGL::PVHBox::pack_end(PVWidget *child, bool expand)
{
	PVLOG_DEBUG("PVGL::PVHBox::%s\n", __FUNCTION__);

	// FIXME! We dont use the end list for now.
	end_list.push_back(PVBoxChild(child,expand));
}

/******************************************************************************
 *
 * PVGL::PVHBox::move
 *
 *****************************************************************************/
void PVGL::PVHBox::move(int x, int y)
{
	PVLOG_DEBUG("PVGL::PVHBox::%s\n", __FUNCTION__);
	allocation.x = x;
	allocation.y = y;
	allocate_size(allocation);
}

