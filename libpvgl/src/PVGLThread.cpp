//! \file PVGLThread.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011
//! Copyright (C) Picviz Labs 2011

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
