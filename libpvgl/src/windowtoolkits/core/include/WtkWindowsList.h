//! \file WindowsList.h
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011

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
