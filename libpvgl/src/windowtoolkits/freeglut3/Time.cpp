/**
 * \file Time.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifdef USE_WTK_FREEGLUT3

#include <GL/freeglut.h>

#include <pvkernel/core/general.h>

#include "../core/include/Time.h"

int PVGL::wtk_time_ms_elapsed_since_init()
{
	return glutGet(GLUT_ELAPSED_TIME);
}

#endif	// USE_WTK_FREEGLUT3
