#include <pvkernel/core/general.h>
#include <gl/simple_lines_int_view.h>
#include <iostream>
#include <tbb/tick_count.h>

void SLIntView::paintGL()
{
	// We start our timer !
	tbb::tick_count start = tbb::tick_count::now();
	std::cout << "Début de l'affichage BZ" << std::endl;
	
	// We clear the Buffers 
	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(1,1,1);

	glBegin(GL_LINES);
	size_t nlines = _pts->size()/4;
	std::cout << "Drawing " << nlines << " lines..." << std::endl;
	for (size_t i = 0; i < nlines-2; i++) {
		glVertex2i((*_pts)[i*4], (*_pts)[i*4+1]);
		glVertex2i((*_pts)[i*4+2], (*_pts)[i*4+3]);
	}
	glColor3f(1,0,0);
	glVertex2i((*_pts)[(nlines-2)*4], (*_pts)[(nlines-2)*4+1]);
	glVertex2i((*_pts)[(nlines-2)*4+2], (*_pts)[(nlines-2)*4+3]);
	glVertex2i((*_pts)[(nlines-1)*4], (*_pts)[(nlines-1)*4+1]);
	glVertex2i((*_pts)[(nlines-1)*4+2], (*_pts)[(nlines-1)*4+3]);
	glEnd();	


	// We stop our time.
	tbb::tick_count end = tbb::tick_count::now();
	std::cout << "Fin de l'affichage BZ: " << (end-start).seconds() << std::endl;
}
