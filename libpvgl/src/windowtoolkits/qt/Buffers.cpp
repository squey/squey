//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011

#ifdef USE_WTK_QT

#include <pvkernel/core/general.h>

#include "../core/include/Buffers.h"

#include "include/global.h"
#include "include/PVGLWidget.h"

int PVGL::wtk_buffers_swap()
{
	PVGL::WTKQt::Global::get_current_window()->swapBuffers();
	return 0;
}

#endif	// USE_WTK_QT
