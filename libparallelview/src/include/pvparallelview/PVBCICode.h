#ifndef PARALLELVIEW_PVBCICODE_H
#define PARALLELVIEW_PVBCICODE_H

#include <stdint.h>
#include <pvkernel/core/general.h>
#include <pvkernel/core/PVAllocators.h>
#include <tbb/cache_aligned_allocator.h>

namespace PVParallelView {

struct PVBCICode
{
	//typedef PVCore::PVAlignedAllocator<PVBCICode, 16> allocator;
	typedef tbb::cache_aligned_allocator<PVBCICode> allocator;
	union {
		uint64_t int_v;
		struct {
			uint32_t idx;
			uint32_t l: 10;
			uint32_t r: 10;
			uint32_t color: 8;
			uint32_t __reserved: 4;
		} s;
	};
	static void init_random_codes(PVBCICode* codes, size_t n);
	static PVBCICode* allocate_codes(size_t n);
	static void free_codes(PVBCICode* codes);
};

typedef PVBCICode DECLARE_ALIGN(16) * PVBCICode_ap;

}

#endif
