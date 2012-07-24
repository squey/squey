/**
 * \file Init.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifdef USE_WTK_QT

#include <pvkernel/core/general.h>

#include "../core/include/Init.h"
#include "include/global.h"

int PVGL::wtk_init(int argc, char **argv)
{
	PVGL::WTKQt::Global::set_ms_start();
	return 0;
}

void PVGL::wtk_main_loop()
{
}

#endif	// USE_WTK_QT
