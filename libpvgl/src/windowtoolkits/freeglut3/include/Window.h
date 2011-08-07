//! \file Window.h
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011

#ifdef USE_WTK_FREEGLUT3

#ifndef LIBPVGL_WTK_FREEGLUT3_WINDOW_H
#define LIBPVGL_WTK_FREEGLUT3_WINDOW_H

namespace PVGL {

	/* How do we access our window? a pointer (QT) or an int (Freeglut)? */
	enum WTK_WINTYPE {
		WINTYPE_INT,
		WINTYPE_POINTER,
	};

	WTK_WINTYPE wtk_window_get_type(void);
	int wtk_window_int_create(const char *name, int width, int height);
	void wtk_window_resize(int width, int height);
	void wtk_window_fullscreen(void);
	void wtk_window_need_redisplay(void);

}

#endif	/* LIBPVGL_WTK_FREEGLUT3_WINDOW_H */

#endif	/* USE_WTK_FREEGLUT3 */
