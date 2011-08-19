//! \file WtkInit.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011

#ifdef USE_WTK_FREEGLUT3
#include <GL/freeglut.h>
#endif //USE_WTK_FREEGLUT3

#include <pvkernel/core/general.h>

#include <WtkInit.h>

static int init_freeglut3(int argc, char **argv)
{

	glutInitContextVersion(3, 3);
	glutInitContextFlags(GLUT_FORWARD_COMPATIBLE | GLUT_DEBUG);
	glutInitContextProfile(GLUT_COMPATIBILITY_PROFILE);
	//  glutInitContextProfile (GLUT_CORE_PROFILE);
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_STENCIL);
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
	PVLOG_INFO("PVGL::%s glut version: %d\n", __FUNCTION__, glutGet(GLUT_VERSION));	

	return 0;		// void glutInit(); so we can only return 0
}

int PVGL::WTK::init(int argc, char **argv)
{
#ifdef USE_WTK_FREEGLUT3
	return init_freeglut3(argc, argv);
#endif //USE_WTK_FREEGLUT3
#ifdef USE_WTK_QT4
	// Nothing to initialize
	return 0;
#endif //USE_WTK_QT4

	return -1;
}
