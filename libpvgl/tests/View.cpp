#include "View.h"
#include <iostream>

void View::paintGL()
{
	glClear(GL_DEPTH_BUFFER_BIT);

	const int fraction = 128;
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
	std::cout << "Fin de l'affichage" << std::endl;
}

void View::set_buffer(Point*p, int size)
{
	buffer = p;
	buffer_size = size;
}

