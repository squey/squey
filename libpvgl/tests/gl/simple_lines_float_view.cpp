/**
 * \file simple_lines_float_view.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <gl/simple_lines_float_view.h>
#include <iostream>
#include <tbb/tick_count.h>

void SLFloatView::paintGL()
{
	// We start our timer !
	tbb::tick_count start = tbb::tick_count::now();
	std::cout << "Début de l'affichage" << std::endl;
	
	// We clear the Buffers 
	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(1,1,1);

	glBegin(GL_LINES);
	size_t nlines = _pts->size()/4;
	for (size_t i = 0; i < nlines; i++) {
		glVertex2f((*_pts)[i*4], (*_pts)[i*4+1]);
		glVertex2f((*_pts)[i*4+2], (*_pts)[i*4+3]);
	}
	glEnd();	


	// We stop our time.
	tbb::tick_count end = tbb::tick_count::now();
	std::cout << "Fin de l'affichage: " << (end-start).seconds() << std::endl;
}
