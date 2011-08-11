//! \file Window.cpp
//! Functions that handle window operations
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011

#ifdef USE_WTK_FREEGLUT3

#include <GL/freeglut.h>

#include <QtGlobal>

#include "include/Window.h"

PVGL::WTK_WINTYPE PVGL::wtk_window_get_type(void)
{
	// We are freeglut, so our window is an int
	return WINTYPE_INT;
}

int PVGL::wtk_window_int_create(const char *name, int width, int height)
{
	glutInitWindowSize(width, height);
	return glutCreateWindow(name);
}

void PVGL::wtk_window_resize(int width, int height)
{
	glutReshapeWindow(width, height);
}

void PVGL::wtk_window_fullscreen()
{
	glutFullScreen();
}

void PVGL::wtk_window_need_redisplay()
{
	glutPostRedisplay();
}

#endif	// USE_WTK_FREEGLUT3
