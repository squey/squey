#include "View.h"
#include <iostream>
#include <tbb/tick_count.h>
void View::paintGL()
{
	glClear(GL_DEPTH_BUFFER_BIT);

	const int fraction = 128;
	tbb::tick_count start = tbb::tick_count::now();
	std::cout << "Début de l'affichage" << std::endl;
	for(int j = 0 ; j < fraction ; j++)
	{
		glBegin(GL_LINES);
		int offset = j*(buffer_size/fraction); 
		for(int i = 0 ; i < buffer_size/fraction ; i++)
		{
			glVertex3d(0, buffer[offset+i].y1, i);
			glVertex3d(1024, buffer[offset+i].y2, i);
		}
		glEnd();
	}
	tbb::tick_count end = tbb::tick_count::now();
	std::cout << "Fin de l'affichage: " << (end-start).seconds() << std::endl;
}

void View::set_buffer(Point*p, int size)
{
	buffer = p;
	buffer_size = size;
}

