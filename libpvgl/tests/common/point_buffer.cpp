/**
 * \file point_buffer.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <common/common.h>
#include <cstdlib>
#include <ctime>
#include <cassert>
#include <cstring>

Point* allocate_buffer(int size)
{
	assert(size > 0 && size%8 == 0);

	Point* table = new Point[size];
	assert(table != NULL);
	
	return table;
}

Point* allocate_buffer_cuda(int size)
{
	assert(size > 0 && size%8 == 0);

	return NULL;
}

void fill_buffer(Point* buffer, int size)
{
	srand(time(NULL));

	for(int i = 0 ; i < size ; i++)
	{
		buffer[i].y1 = rand() % PIXELS_CB;
		buffer[i].y2 = rand() % PIXELS_CB;
	}
}