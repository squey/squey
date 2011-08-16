//! \file LoadingFunction.h
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011

#ifdef USE_WTK_FREEGLUT3

#ifndef LIBPVGL_WTK_FREEGLUT3_LOADINGFUNCTION_H
#define LIBPVGL_WTK_FREEGLUT3_LOADINGFUNCTION_H

#include <pvgl/PVWidgetManager.h>

namespace PVGL {

	int wtk_loading_function(PVWidgetManager &widget_manager, int width, int height);

}

#endif	/* LIBPVGL_WTK_FREEGLUT3_LOADINGFUNCTION_H */

#endif	/* USE_WTK_FREEGLUT3 */
