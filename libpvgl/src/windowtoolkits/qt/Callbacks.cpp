/**
 * \file Callbacks.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifdef USE_WTK_QT

#include "../core/include/Callbacks.h"
#include "include/global.h"

// For usleep/Sleep
#ifdef WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <boost/thread.hpp>
#include <boost/bind.hpp>

void PVGL::wtk_set_close_func(pvgl_callback_close_t f)
{
	PVGL::WTKQt::Global::_callback_close = f;
}

void PVGL::wtk_set_display_func(pvgl_callback_display_t f)
{
	PVGL::WTKQt::Global::_callback_display = f;
}

void PVGL::wtk_set_entry_func(pvgl_callback_entry_t f)
{
	PVGL::WTKQt::Global::_callback_entry = f;
}

void PVGL::wtk_set_idle_func(pvgl_callback_idle_t f)
{
	PVGL::WTKQt::Global::_callback_idle = f;
}

void PVGL::wtk_set_keyboard_func(pvgl_callback_keyboard_t f)
{
	PVGL::WTKQt::Global::_callback_keyboard = f;
}

void PVGL::wtk_set_motion_func(pvgl_callback_motion_t f)
{
	PVGL::WTKQt::Global::_callback_motion = f;
}

void PVGL::wtk_set_mouse_func(pvgl_callback_mouse_t f)
{
	PVGL::WTKQt::Global::_callback_mouse = f;
}

void PVGL::wtk_set_passive_motion_func(pvgl_callback_passive_motion_t f)
{
	PVGL::WTKQt::Global::_callback_passive_motion = f;
}

void PVGL::wtk_set_reshape_func(pvgl_callback_reshape_t f)
{
	PVGL::WTKQt::Global::_callback_reshape = f;
}

void PVGL::wtk_set_special_func(pvgl_callback_special_t f)
{
	PVGL::WTKQt::Global::_callback_special = f;
}

// Timers
static void timer_thread(unsigned int msecs, PVGL::pvgl_callback_timer_t f, int value)
{
#ifdef WIN32
	Sleep(msecs);
#else
	useconds_t usecs = msecs*1000;
	usleep(usecs);
#endif
	f(value);
}

void PVGL::wtk_set_timer_func(unsigned int msecs, pvgl_callback_timer_t f, int value)
{
	//boost::thread thread(boost::bind(timer_thread, msecs, f, value));
}


#endif	// USE_WTK_QT
