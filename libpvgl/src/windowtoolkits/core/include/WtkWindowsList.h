/**
 * \file WtkWindowsList.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef LIBPVGL_WTK_WINDOWSLIST_H
#define LIBPVGL_WTK_WINDOWSLIST_H

#include <list>

#include "WtkWindow.h"

namespace PVGL {
	namespace WTK {

	class WtkWindowsList {
	private:
		std::list<WtkWindow*> _windows_list;
	public:
		WtkWindowsList();
	};

	}
}

#endif	/* LIBPVGL_WTK_WINDOWSLIST_H */
