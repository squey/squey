#include <pvkernel/core/general.h>
#include <pvparallelview/simple_lines_int_view.h>
#include <iostream>
#include <tbb/tick_count.h>

void SLIntView::paintGL()
{
	
	// We clear the Buffers 
	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(1,1,1);

	tbb::tick_count start = tbb::tick_count::now();
	glBegin(GL_LINES);
	size_t nlines = _pts->size()/4;
	std::cout << "Drawing " << nlines << " lines..." << std::endl;
	for (size_t i = 0; i < nlines; i++) {
		if (_colors) {
			PVRGB const& rgb((*_colors)[i]);
			glColor3ub(rgb.s.r, rgb.s.g, rgb.s.b);
		}
		glVertex2i((*_pts)[i*4], (*_pts)[i*4+1]);
		glVertex2i((*_pts)[i*4+2], (*_pts)[i*4+3]);
	}
	/*glVertex2i((*_pts)[(nlines-2)*4], (*_pts)[(nlines-2)*4+1]);
	glVertex2i((*_pts)[(nlines-2)*4+2], (*_pts)[(nlines-2)*4+3]);
	glVertex2i((*_pts)[(nlines-1)*4], (*_pts)[(nlines-1)*4+1]);
	glVertex2i((*_pts)[(nlines-1)*4+2], (*_pts)[(nlines-1)*4+3]);*/
	glEnd();	

	glFinish();
	tbb::tick_count end = tbb::tick_count::now();
	double dur = (end-start).seconds();
	printf("Drawing duration: %0.4f ms (%0.4f lines/s)\n", dur*1000.0, (double)(nlines)/dur);
}
