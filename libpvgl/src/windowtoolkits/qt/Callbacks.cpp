//! \file Callbacks.cpp
//! Functions that handle window operations
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011

#ifdef USE_WTK_QT

#include <GL/freeglut.h>

#include "../core/include/Callbacks.h"
#include "include/global.h"

void PVGL::wtk_set_close_func(pvgl_callback_close_t f)
{
	_callback_close = f;
}

void PVGL::wtk_set_display_func(pvgl_callback_display_t f)
{
	_callback_display = f;
}

void PVGL::wtk_set_entry_func(pvgl_callback_entry_t f)
{
	_callback_entry = f;
}

void PVGL::wtk_set_idle_func(pvgl_callback_idle_t f)
{
	_callback_idle = f;
}

void PVGL::wtk_set_keyboard_func(pvgl_callback_keyboard_t f)
{
	_callback_keyboard = f;
}

void PVGL::wtk_set_motion_func(pvgl_callback_motion_t f)
{
	_callback_motion = f;
}

void PVGL::wtk_set_mouse_func(pvgl_callback_mouse_t f)
{
	_callback_mouse = f;
}

void PVGL::wtk_set_passive_motion_func(pvgl_callback_passive_motion_t f)
{
	_callback_passive_motion = f;
}

void PVGL::wtk_set_reshape_func(pvgl_callback_reshape_t f)
{
	_callback_reshape = f;
}

void PVGL::wtk_set_special_func(pvgl_callback_special_t f)
{
	_callback_special = f;
}

void PVGL::wtk_set_timer_func(unsigned int msecs, pvgl_callback_timer_t f, int value)
{
	PVGL::WTKQt::Global::launch_timer(msecs, f, value);
}


#endif	// USE_WTK_QT
