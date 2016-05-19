/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PARALLELVIEW_PVBCICODE_H
#define PARALLELVIEW_PVBCICODE_H

#include <stdint.h>
//#include <pvkernel/core/PVAllocators.h>
#include <pvkernel/core/PVHSVColor.h>
#include <pvparallelview/common.h>
#include <tbb/cache_aligned_allocator.h>

namespace PVParallelView
{

/**
 * It represents a line in a Zone of ParallelView
 *
 * * Line is:
 *     * position (listing position)
 *     * left plotting value
 *     * right plotting value
 *     * color
 *     * direction (use?)
 */
template <size_t Bbits = NBITS_INDEX>
struct PVBCICode {
#ifndef __CUDACC__
	static_assert((Bbits >= 1) & (Bbits <= 11), "PVBCICode: Bbits must be between 1 and 11.");
#endif

	typedef tbb::cache_aligned_allocator<PVBCICode> allocator;
	union {
		uint64_t int_v;
		struct {
			uint32_t idx;
			uint32_t l : Bbits;
			uint32_t r : Bbits;
			uint32_t color : 8;
			uint32_t type : 2;
		} __attribute((packed)) s;
	};

	typedef enum { STRAIGHT = 0, UP = 1, DOWN = 2 } _type_t;

	static PVBCICode* allocate_codes(size_t n)
	{
		PVBCICode* ret = PVBCICode::allocator().allocate(n);
		return ret;
	}

	static void free_codes(PVBCICode* codes) { PVBCICode::allocator().deallocate(codes, 0); }
};

typedef union {
	PVBCICode<10> as_10;
	PVBCICode<11> as_11;
} PVBCICodeBase;
}

#endif
