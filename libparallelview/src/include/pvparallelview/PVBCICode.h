/**
 * \file PVBCICode.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PARALLELVIEW_PVBCICODE_H
#define PARALLELVIEW_PVBCICODE_H

#include <stdint.h>
#include <pvkernel/core/general.h>
#include <pvkernel/core/PVAllocators.h>
#include <pvkernel/core/PVHSVColor.h>
#include <pvparallelview/common.h>
#include <tbb/cache_aligned_allocator.h>

namespace PVParallelView {

template <size_t Bbits = NBITS_INDEX>
struct PVBCICode
{
#ifndef __CUDACC__
	static_assert((Bbits >= 1) & (Bbits <= 11), "PVBCICode: Bbits must be between 1 and 11.");
#endif

	typedef PVBCICode<Bbits> DECLARE_ALIGN(16) ap_t;
	typedef uint64_t int_type;

	//typedef PVCore::PVAlignedAllocator<PVBCICode, 16> allocator;
	typedef tbb::cache_aligned_allocator<PVBCICode> allocator;
	union {
		uint64_t int_v;
		struct {
			uint32_t idx;
			uint32_t l: Bbits;
			uint32_t r: Bbits;
			uint32_t color: 8;
			uint32_t type: 2;
		} s;
	};

	typedef enum {
		STRAIGHT = 0,
		UP = 1,
		DOWN = 2
	} _type_t;

	static void init_random_codes(PVBCICode* codes, size_t n)
	{
		for (size_t i = 0; i < n; i++) {
			PVBCICode c;
			c.int_v = 0;
			//c.s.idx = rand();
			c.s.idx = n-i;
			//c.s.l = ((i/1024)*4)%1024;
			//c.s.l = i&(MASK_INT_YCOORD);
			//c.s.l = rand()&constants<Bbits>::mask_int_ycoord;
			//c.s.r = rand()&constants<Bbits>::mask_int_ycoord;
			//c.s.r = (c.s.l+10)&constants<Bbits>::mask_int_ycoord;
			if (i < 1024) {
				c.s.l = constants<Bbits>::mask_int_ycoord/2;
				c.s.type = UP;
				c.s.color = HSV_COLOR_WHITE;
			}
			else 
			if (i < 3072) {
				c.s.l = constants<Bbits>::mask_int_ycoord/2;
				c.s.type = DOWN;
				c.s.color = HSV_COLOR_BLACK;
			}
			else {
				c.s.l = constants<Bbits>::mask_int_ycoord/5;
				c.s.color = i%((1<<HSV_COLOR_NBITS_ZONE)*6);
			}
			c.s.r = i&(constants<Bbits>::mask_int_ycoord);
			//c.s.color = rand()&((1<<9)-1);
			//c.s.color = i%((1<<HSV_COLOR_NBITS_ZONE)*6);
			codes[i] = c;
		}
	}

	static PVBCICode* allocate_codes(size_t n)
	{
		PVBCICode* ret = PVBCICode::allocator().allocate(n);
		return ret;
	}

	static void free_codes(PVBCICode* codes)
	{
		PVBCICode::allocator().deallocate(codes, 0);
	}
};

}

#endif
