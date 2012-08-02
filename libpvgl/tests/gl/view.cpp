/**
 * \file view.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <gl/view.h>
#include <iostream>
#include <tbb/tick_count.h>




View::View(QWidget *parent)
	: QGLWidget(parent)
{
	buffer_size = 0;
}

void View::compute_collision_table() {
	// The main loop
	for (int i = 0 ; i < buffer_size ; i++)
	{		
		int y1 = buffer[i].y1;
		int y2 = buffer[i].y2;
		int key = 1024*y1 + y2;
		
		// We test if the line is already present in the collision_table
		if (ugly_collision_table[key] == 0) {
			// The line is new to our drawing!
			// We must leave a trace in the collision_table
			ugly_collision_table[key] = 1;
		}
	}

}

void View::paintGL()
{
	// We start our timer !
	tbb::tick_count start = tbb::tick_count::now();
	std::cout << "Début de l'affichage" << std::endl;
	
	// We clear the Buffers 
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
        //glClear(GL_COLOR_BUFFER_BIT);
	
	// We reset the collision_table
	//reset_collision_table();	
	
	// We prepare our batches' loop
	const int fraction = 128;

	glColor3f(1,1,1);
/*	// The main loop
	for(int j = 0 ; j < fraction ; j++)
	{
		glBegin(GL_LINES);
		int offset = j*(buffer_size/fraction); 
		for(int i = 0 ; i < buffer_size/fraction ; i++)
		{
			int mon_i = offset+i;
			int y1 = buffer[mon_i].y1;
			int y2 = buffer[mon_i].y2;
			int key = 1024*y1 + y2;
			
			// We test if the line is already present in the collision_table
			if (ugly_collision_table[key] != 0) {
				// The line is new to our drawing!
				glVertex3i(0, y1, 0);
				glVertex3i(1024, y2, 0);
				// We must leave a trace in the collision_table
				//ugly_collision_table[key] = 1;
			}
		}
		glEnd();
	}*/

	glBegin(GL_LINES);
	for (int i=0; i<1024*1024 ; i++) {
		if ( ugly_collision_table[i] != 0) {
			int y1 = i/1024;
			int y2 = i % 1024;
			glVertex3i(0, y1, 0);
			glVertex3i(1024, y2, 0);
		}
	}
	glEnd();	


	// We stop our time.
	tbb::tick_count end = tbb::tick_count::now();
	std::cout << "Fin de l'affichage: " << (end-start).seconds() << std::endl;
}


void View::reset_collision_table()
{
	// We erase the ugly_collision_table
	for (int i=0; i<1024*1024; i++) {
		ugly_collision_table[i] = 0;
	}
}


void View::set_buffer(Point*p, int size)
{
	buffer = p;
	buffer_size = size;
	
	reset_collision_table();
	// now we can compute the collision_table
	compute_collision_table();
}

