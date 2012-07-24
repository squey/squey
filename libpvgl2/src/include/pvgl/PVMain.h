/**
 * \file PVMain.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef LIBPVGL_PVMAIN_H
#define LIBPVGL_PVMAIN_H

#include <pvkernel/core/general.h>

namespace PVGL {

struct LibGLDecl PVMain {

	/**
	*
	*/
	static void timer_func(int);

	/**
	*
	*/
	static void stop();
};

} /* namespace PVGL */

#endif	/* LIBPVGL_PVMAIN_H */

