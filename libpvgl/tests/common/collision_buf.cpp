#include <common/common.h>
#include <cstdlib>
#include <cstring>
#include <cassert>

#include <tbb/cache_aligned_allocator.h>

CollisionBuffer allocate_CB()
{
	//return new int[1024*1024/8];
	static tbb::cache_aligned_allocator<int> alloc;
	CollisionBuffer ret;
	ret = alloc.allocate(NB_INT_CB);
	assert((uintptr_t) ret % 16 == 0); 
	//posix_memalign((void**) &ret, 16, SIZE_CB);

	memset(ret, 0, SIZE_CB);

	return ret;
}

void free_CB(CollisionBuffer b)
{
	static tbb::cache_aligned_allocator<int> alloc;
	alloc.deallocate(b, NB_INT_CB);
	//free(b);
}
