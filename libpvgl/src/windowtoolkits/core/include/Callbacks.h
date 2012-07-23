/**
 * \file Callbacks.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef LIBPVGL_WTK_CALLBACKS_H
#define LIBPVGL_WTK_CALLBACKS_H

namespace PVGL {

	typedef void(*pvgl_callback_close_t)();
	typedef void(*pvgl_callback_display_t)();
	typedef void(*pvgl_callback_entry_t)(int);
	typedef void(*pvgl_callback_idle_t)();
	typedef void(*pvgl_callback_keyboard_t)(unsigned char, int, int);
	typedef void(*pvgl_callback_motion_t)(int, int);
	typedef void(*pvgl_callback_passive_motion_t)(int, int);
	typedef void(*pvgl_callback_mouse_t)(int, int, int, int);
	typedef void(*pvgl_callback_reshape_t)(int, int);
	typedef void(*pvgl_callback_special_t)(int, int, int);
	typedef void(*pvgl_callback_timer_t)(int);

	void wtk_set_close_func(pvgl_callback_close_t f);
	void wtk_set_display_func(pvgl_callback_display_t f);
	void wtk_set_entry_func(pvgl_callback_entry_t f);
	void wtk_set_idle_func(pvgl_callback_idle_t f);
	void wtk_set_keyboard_func(pvgl_callback_keyboard_t f);
	void wtk_set_motion_func(pvgl_callback_motion_t f);
	void wtk_set_mouse_func(pvgl_callback_mouse_t f);
	void wtk_set_passive_motion_func(pvgl_callback_passive_motion_t f);
	void wtk_set_reshape_func(pvgl_callback_reshape_t f);
	void wtk_set_special_func(pvgl_callback_special_t f);
	void wtk_set_timer_func(unsigned int msecs, pvgl_callback_timer_t f, int value);

}

#endif
