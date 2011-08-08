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
		enum WTK_WINDOWTYPE {
			WTK_WINDOWTYPE_INT,
			WTK_WINDOWTYPE_POINTER,
		};

		int   _win_id;
		void *_win_ptr;

		WTK_WINDOWTYPE _win_type;
	public:
		WtkWindow(int win_id);
		WtkWindow(void *win_ptr);
	};

	}
}

#endif	/* LIBPVGL_WTK_WINDOW_H */
