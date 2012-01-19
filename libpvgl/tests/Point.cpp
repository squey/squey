#include "Point.h"
#include <cstdlib>
#include <ctime>
#include <cassert>
#include <cstring>

#include <tbb/cache_aligned_allocator.h>

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
		buffer[i].y1 = 3 % 1024;
		buffer[i].y2 = 2 % 1024;
	}
}

CollisionBuffer allocate_CB()
{
	//return new int[1024*1024/8];
	static tbb::cache_aligned_allocator<int> alloc;
	CollisionBuffer ret;
	/*ret = alloc.allocate(NB_INT_CB);
	assert((uintptr_t) ret % 16 == 0); */
	posix_memalign((void**) &ret, 16, SIZE_CB);

	memset(ret, 0, SIZE_CB);

	return ret;
}

void free_CB(CollisionBuffer b)
{
	//static tbb::cache_aligned_allocator<int> alloc;
	//alloc.deallocate(b, NB_INT_CB);
	free(b);
}
