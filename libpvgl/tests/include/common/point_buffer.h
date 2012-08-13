/**
 * \file point_buffer.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef point_h
#define point_h

struct Point
{
	int y1;
	int y2;
};

Point* allocate_buffer(int size);
Point* allocate_buffer_cuda(int size);
void fill_buffer(Point* buffer, int size);

#endif
