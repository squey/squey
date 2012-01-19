#include "View.h"
#include <iostream>

void View::paintGL()
{
	std::cout << "Début de l'affichage" << std::endl;
	glBegin(GL_LINES);
	for(int i = 0 ; i < buffer_size ; i++)
	{
		glVertex2d(0, buffer[i].y1);
		glVertex2d(1024, buffer[i].y2);
	}
	glEnd();
	std::cout << "Fin de l'affichage" << std::endl;
}

void View::set_buffer(Point*p, int size)
{
	buffer = p;
	buffer_size = size;
}

