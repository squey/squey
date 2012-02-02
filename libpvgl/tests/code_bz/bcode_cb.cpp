#include <code_bz/bcode_cb.h>
#include <code_bz/types.h>
#include <cstdlib>
#include <string.h>

#include <tbb/cache_aligned_allocator.h>

BCodeCB allocate_BCodeCB()
{
	//static tbb::cache_aligned_allocator<int> alloc;
	BCodeCB ret;
	posix_memalign((void**) &ret, 16, SIZE_BCODECB);
	//ret = (BCodeCB) alloc.allocate(NB_INT_BCODECB);
	memset(ret, 0, SIZE_BCODECB);
	return ret;
}

void free_BCodeCB(BCodeCB cb)
{
	//static tbb::cache_aligned_allocator<int> alloc;
	//alloc.deallocate((int*) cb, NB_INT_BCODECB);
	free(cb);
}

void bcode_cb_to_bcodes(std::vector<PVBCode>& ret, BCodeCB cb)
{
	for (size_t i = 0; i < NB_INT_BCODECB; i++) {
		uint32_t tmp = cb[i];
		for (int j = 0; j < 32; j++) {
			if ((tmp & (1<<j)) != 0) {
				PVBCode code;
				code.int_v = (i<<5) + j;
				ret.push_back(code);
			}
		}
	}
}
