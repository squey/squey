//! \file Window.cpp
//! Functions that handle window operations
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011

#ifdef USE_WTK_FREEGLUT3

#define GLEW_STATIC 1
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "include/Window.h"

void PVGL::wtk_window_resize(int width, int height)
{
	glutReshapeWindow(width, height);
}

void PVGL::wtk_window_fullscreen()
{
	glutFullScreen();
}

#endif	// USE_WTK_FREEGLUT3
