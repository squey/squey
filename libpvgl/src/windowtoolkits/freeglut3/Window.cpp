/**
 * \file Window.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifdef USE_WTK_FREEGLUT3

#include <GL/freeglut.h>

#include <QtGlobal>

#include "../core/include/Window.h"

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

int PVGL::wtk_get_current_window()
{
	return glutGetWindow();
}

void PVGL::wtk_set_current_window(int id)
{
	glutSetWindow(id);
}

void PVGL::wtk_destroy_window(int id)
{
	glutDestroyWindow(id);
}

int PVGL::wtk_get_keyboard_modifiers()
{
	return glutGetModifiers();
}

#endif	// USE_WTK_FREEGLUT3
