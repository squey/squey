#include "Point.h"
#include <cstdlib>
#include <ctime>

Point* allocate_buffer(int size)
{
	assert(size > 0 && size%8 == 0);

	Point* table = new Point[size];
	assert(table != NULL);
	
	return table;
}

void fill_buffer(Point* buffer, int size)
{
	srand(time(NULL));

	for(int i = 0 ; i < size ; i++)
	{
		buffer[i].y1 = rand() % 1024;
		buffer[i].y2 = rand() % 1024;
	}
}

CollisionBuffer allocate_CB()
{
	return new int[1024*1024/8];
}
