/**
 * \file PVMisc.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <pvgl/PVWidgetManager.h>
#include <pvgl/PVContainer.h>

#include <pvgl/PVMisc.h>

/******************************************************************************
 *
 * PVGL::PVMisc::PVMisc
 *
 *****************************************************************************/
PVGL::PVMisc::PVMisc(PVWidgetManager *widget_manager_) :
PVWidget(widget_manager_)
{
	PVLOG_DEBUG("PVGL::PVMisc::%s\n", __FUNCTION__);

	align_x = align_y = 0.5;
	padding_x = padding_y = 0;
}

/******************************************************************************
 *
 * PVGL::PVMisc::set_alignment
 *
 *****************************************************************************/
void PVGL::PVMisc::set_alignment(float x, float y)
{
	PVLOG_DEBUG("PVGL::PVMisc::%s\n", __FUNCTION__);

	align_x = picviz_min(picviz_max(x, 0.0f), 1.0f);
	align_y = picviz_min(picviz_max(y, 0.0f), 1.0f);
}

/******************************************************************************
 *
 * PVGL::PVMisc::set_padding
 *
 *****************************************************************************/
void PVGL::PVMisc::set_padding(int x, int y)
{
	PVLOG_DEBUG("PVGL::PVMisc::%s\n", __FUNCTION__);

  x = picviz_max(x, 0);
	y = picviz_max(y, 0);

	if (x != padding_x || y != padding_y) {
		requisition.width -= 2 * padding_x;
		requisition.height-= 2 * padding_y;
		padding_x = x;
		padding_y = y;
		requisition.width += 2 * padding_x;
		requisition.height+= 2 * padding_y;
		if (parent)
			parent->size_adjust();
	}
}

