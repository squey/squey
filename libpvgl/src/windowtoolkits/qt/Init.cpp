//! \file LoadingFunction.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011

#ifdef USE_WTK_FREEGLUT3

#include <pvkernel/core/general.h>

#include "../core/include/Init.h"

int PVGL::wtk_init(int argc, char **argv)
{
	WTKQt::Global::set_ms_start();
	return 0;
}

void PVGL::wtk_main_loop()
{
}

#endif	// USE_WTK_FREEGLUT3
