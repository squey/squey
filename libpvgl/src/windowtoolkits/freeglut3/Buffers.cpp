/**
 * \file Buffers.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifdef USE_WTK_FREEGLUT3

#include <GL/freeglut.h>

#include <pvkernel/core/general.h>

#include "../core/include/Buffers.h"

int PVGL::wtk_buffers_swap()
{
	//static int frequency = 0;frequency++;if(frequency%4==0)
	glutSwapBuffers();
	return 0;
}

#endif	// USE_WTK_FREEGLUT3
