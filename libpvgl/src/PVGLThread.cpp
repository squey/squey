/**
 * \file PVGLThread.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <pvgl/general.h>
#include <pvgl/PVGLThread.h>

/******************************************************************************
 *
 * PVGL::PVGLThread::run
 *
 *****************************************************************************/
void PVGL::PVGLThread::run ()
{
	PVLOG_DEBUG("PVGL::PVThread::%s\n", __FUNCTION__);

	pvgl_init(pvsdk_messenger);
}
