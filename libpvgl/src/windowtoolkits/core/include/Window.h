//! \file Window.h
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011

#ifndef LIBPVGL_WTK_WINDOW_H
#define LIBPVGL_WTK_WINDOW_H

namespace PVGL {

	int wtk_window_int_create(const char *name, int width, int height);
	void wtk_window_resize(int width, int height);
	void wtk_window_fullscreen(void);
	void wtk_window_need_redisplay(void);
	int wtk_get_current_window();
	void wtk_set_current_window(int id);
	void wtk_destroy_window(int id);
	int wtk_get_keyboard_modifiers();

}

#endif	/* LIBPVGL_WTK_WINDOW_H */
