#ifndef point_h
#define point_h

struct Point
{
	int y1;
	int y2;
};

#define DECLARE_ALIGN(n) __attribute__((aligned(n)))
w
typedef int* DECLARE_ALIGN(16) CollisionBuffer;

Point* allocate_buffer(int size);
Point* allocate_buffer_cuda(int size);

CollisionBuffer allocate_CB(void);

#define NB_INT_CB (1024*1024/32)
#define SIZE_CB (NB_INT_CB*sizeof(int))

#endif
