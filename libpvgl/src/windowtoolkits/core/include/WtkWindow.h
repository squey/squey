//! \file Window.h
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011

#ifndef LIBPVGL_WTK_WINDOW_H
#define LIBPVGL_WTK_WINDOW_H

namespace PVGL {
	namespace WTK {

	class WtkWindow {
	private:
		int   _win_id;
		void *_win_ptr;
	public:
		WtkWindow(int win_id);
		WtkWindow(void *win_ptr);
		enum WTK_WINDOWTYPE {
			WINDOWTYPE_INT,
			WINDOWTYPE_POINTER,
		};

	};

	}
}

#endif	/* LIBPVGL_WTK_WINDOW_H */
