#ifndef point_h
#define point_h

struct Point
{
	int y1;
	int y2;
};

typedef int* CollisionBuffer;

Point* allocate_buffer(int size);
Point* allocate_buffer_cuda(int size);

CollisionBuffer allocate_CB(void);

#endif
