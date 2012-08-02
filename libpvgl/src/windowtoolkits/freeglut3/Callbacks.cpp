/**
 * \file Callbacks.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifdef USE_WTK_FREEGLUT3

#include <GL/freeglut.h>

#include <QtGlobal>

#include "../core/include/Callbacks.h"

void PVGL::wtk_set_close_func(pvgl_callback_close_t f)
{
	glutCloseFunc(f);
}

void PVGL::wtk_set_display_func(pvgl_callback_display_t f)
{
	glutDisplayFunc(f);
}

void PVGL::wtk_set_entry_func(pvgl_callback_entry_t f)
{
	glutEntryFunc(f);
}

void PVGL::wtk_set_idle_func(pvgl_callback_idle_t f)
{
	glutIdleFunc(f);
}

void PVGL::wtk_set_keyboard_func(pvgl_callback_keyboard_t f)
{
	glutKeyboardFunc(f);
}

void PVGL::wtk_set_motion_func(pvgl_callback_motion_t f)
{
	glutMotionFunc(f);
}

void PVGL::wtk_set_mouse_func(pvgl_callback_mouse_t f)
{
	glutMouseFunc(f);
}

void PVGL::wtk_set_passive_motion_func(pvgl_callback_passive_motion_t f)
{
	glutPassiveMotionFunc(f);
}

void PVGL::wtk_set_reshape_func(pvgl_callback_reshape_t f)
{
	glutReshapeFunc(f);
}

void PVGL::wtk_set_special_func(pvgl_callback_special_t f)
{
	glutSpecialFunc(f);
}

void PVGL::wtk_set_timer_func(unsigned int msecs, pvgl_callback_timer_t f, int value)
{
	glutTimerFunc(msecs, f, value);
}


#endif	// USE_WTK_FREEGLUT3
