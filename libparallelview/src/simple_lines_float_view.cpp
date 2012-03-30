#include <pvparallelview/simple_lines_float_view.h>
#include <iostream>
#include <tbb/tick_count.h>

#define TILESIZE (1024*10)
void SLFloatView::paintGL()
{
	// We start our timer !
	tbb::tick_count start = tbb::tick_count::now();
	std::cout << "Début de l'affichage" << std::endl;
	
	// We clear the Buffers 
	//glClearColor(1,1,0,1);
	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(1,1,1);

	size_t nlines = _pts->size()/4;
	for (size_t t = 0; t < (nlines/TILESIZE)*TILESIZE; t += TILESIZE) {
		glBegin(GL_LINES);
		for (size_t i = t; i < t+TILESIZE; i++) {
			glVertex2f((*_pts)[i*4], (*_pts)[i*4+1]);
			glVertex2f((*_pts)[i*4+2], (*_pts)[i*4+3]);
		}
		glEnd();
	}

	glBegin(GL_LINES);
	for (size_t i = (nlines/TILESIZE)*TILESIZE; i < nlines; i++) {
		glVertex2f((*_pts)[i*4], (*_pts)[i*4+1]);
		glVertex2f((*_pts)[i*4+2], (*_pts)[i*4+3]);
	}
	glEnd();

	// We stop our time.
	tbb::tick_count end = tbb::tick_count::now();
	std::cout << "Fin de l'affichage: " << (end-start).seconds() << std::endl;
}
