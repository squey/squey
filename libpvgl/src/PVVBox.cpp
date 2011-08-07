//! \file PVVBox.cpp
//! $Id: PVVBox.cpp 2520 2011-04-30 12:26:48Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <pvgl/PVWidgetManager.h>

#include <pvgl/PVVBox.h>

/******************************************************************************
 *
 * PVGL::PVVBox::PVVBox
 *
 *****************************************************************************/
PVGL::PVVBox::PVVBox(PVWidgetManager *widget_manager_) :
PVBox(widget_manager_)
{
	PVLOG_DEBUG("PVGL::PVVBox::%s\n", __FUNCTION__);

}

/******************************************************************************
 *
 * PVGL::PVVBox::allocate_size
 *
 *****************************************************************************/
void PVGL::PVVBox::allocate_size(const PVAllocation &new_allocation)
{
	int requisition_height = 0;
	int min_requisition_height = 0;
	int children_number = 0;
	int expandable_number = 0;
	int shrinkable_children = 0;
	int extra_amount;
	int min_extra_amount;
  PVAllocation child_allocation;

	PVLOG_HEAVYDEBUG("PVGLVbox::%s\n", __FUNCTION__);

	allocation = new_allocation;
	for (std::list<PVBoxChild>::iterator it = start_list.begin(); it != start_list.end(); ++it) {
		if (it->child->is_visible()) {
			requisition_height += it->child->get_requisition().height;
			if (it->child->min_size.height != -1) {
				shrinkable_children++;
				min_requisition_height += it->child->min_size.height;
			} else {
				min_requisition_height += it->child->get_requisition().height;
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
	extra_amount = new_allocation.height - requisition.height - 2 * border_width;
	min_extra_amount = new_allocation.height - min_requisition_height - 2 * border_width;

	child_allocation.x = new_allocation.x + border_width;
	child_allocation.y = new_allocation.y + border_width;
	child_allocation.width = new_allocation.width - 2 * border_width;

	if (extra_amount < 0) {
		for (std::list<PVBoxChild>::iterator it = start_list.begin(); it != start_list.end(); ++it) {
			if (it->child->is_visible()) {
				if (it->child->min_size.height != -1) {
					int extra = min_extra_amount / shrinkable_children;
					child_allocation.height = it->child->min_size.height + extra;
					min_extra_amount -= extra;
					shrinkable_children--;
				} else {
					child_allocation.height = it->child->get_requisition().height;
				}
				it->child->allocate_size(child_allocation);
				child_allocation.y += child_allocation.height;
			}
		}
	} else {
		for (std::list<PVBoxChild>::iterator it = start_list.begin(); it != start_list.end(); ++it) {
			int extra;

			if (expandable_number) {
				extra	= extra_amount / children_number;
			} else {
				extra = 0;
			}

			if (it->child->is_visible()) {
				if (it->expand) {
					child_allocation.height = it->child->get_requisition().height + extra;
					it->child->allocate_size(child_allocation);
					child_allocation.y += child_allocation.height;
					extra_amount -= extra;
					expandable_number--;
				} else {
					child_allocation.height = it->child->get_requisition().height;
					it->child->allocate_size(child_allocation);
					child_allocation.y += child_allocation.height;
					children_number--;
				}
			}
		}
	}
}

/******************************************************************************
 *
 * PVGL::PVVBox::size_adjust
 *
 *****************************************************************************/
void PVGL::PVVBox::size_adjust()
{
	PVLOG_HEAVYDEBUG("PVGLVbox::%s\n", __FUNCTION__);

  requisition.width = 2 * border_width;
	requisition.height= 2 * border_width;

	for (std::list<PVBoxChild>::iterator it = start_list.begin(); it != start_list.end(); ++it) {
		if (it->child->is_visible()) {
			requisition.height += it->child->get_requisition().height;
			requisition.width = std::max(requisition.width, 2 * border_width + it->child->get_requisition().width);
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
 * PVGL::PVVBox::pack_start
 *
 *****************************************************************************/
void PVGL::PVVBox::pack_start(PVWidget *child, bool expand)
{
	PVLOG_DEBUG("PVGL::PVVBox::%s\n", __FUNCTION__);

	if (child->is_visible()) {
		if (!start_list.empty()) {
			requisition.height += child->get_requisition().height;
			requisition.width = std::max(requisition.width, 2 * border_width + child->get_requisition().width);
		} else {
			requisition.width = 2 *border_width + child->get_requisition().width;
			requisition.height= 2 *border_width + child->get_requisition().height;
		}
	}
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
 * PVGL::PVVBox::pack_end
 *
 *****************************************************************************/
void PVGL::PVVBox::pack_end(PVWidget *child, bool expand)
{
	PVLOG_DEBUG("PVGL::PVVBox::%s\n", __FUNCTION__);

	end_list.push_back(PVBoxChild(child, expand));
}

/******************************************************************************
 *
 * PVGL::PVVBox::move
 *
 *****************************************************************************/
void PVGL::PVVBox::move(int x, int y)
{
	PVLOG_DEBUG("PVGL::PVVBox::%s\n", __FUNCTION__);
	allocation.x = x;
	allocation.y = y;
	allocate_size(allocation);
}

