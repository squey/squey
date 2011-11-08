//! \file LoadingFunction.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011

#ifdef USE_WTK_FREEGLUT3


#define GLEW_STATIC 1
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <pvgl/PVConfig.h>

#include "../core/include/LoadingFunction.h"

int PVGL::wtk_loading_function(PVWidgetManager &widget_manager, int width, int height)
{
	int current_time = (glutGet(GLUT_ELAPSED_TIME) / 250) % 4;
	const char *text = "";
	switch (current_time) {
	case 0:
		text = "Loading";
		break;
	case 1:
		text = "Loading.";
		break;
	case 2:
		text = "Loading..";
		break;
	case 3:
		text = "Loading...";
		break;
	}
	glOrtho(0, width, height, 0, -1,1);
	
	glColor4ubv(&PVGL_VIEW_LOADING_COLOR.x);
	widget_manager.draw_text(50, 50, text, 22);
	
	return 0;
}

#endif	// USE_WTK_FREEGLUT3
