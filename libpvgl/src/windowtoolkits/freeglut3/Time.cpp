//! \file Buffer.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011

#ifdef USE_WTK_FREEGLUT3

#include <GL/freeglut.h>

#include <pvkernel/core/general.h>

#include "../core/include/Time.h"

int PVGL::wtk_time_ms_elapsed_since_init()
{
	return glutGet(GLUT_ELAPSED_TIME);
}

#endif	// USE_WTK_FREEGLUT3
