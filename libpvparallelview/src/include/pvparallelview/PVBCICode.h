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

class PVBCICodeBase;

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

	static PVBCICode* allocate_codes(size_t n)
	{
		PVBCICode* ret = PVBCICode::allocator().allocate(n);
		return ret;
	}

	static void free_codes(PVBCICode* codes)
	{
		PVBCICode::allocator().deallocate(codes, 0);
	}

	operator PVBCICodeBase& ()
	{
		return *((PVBCICodeBase*)this);
	}

	operator PVBCICodeBase const& ()
	{
		return *((PVBCICodeBase*)this);
	}
};

struct PVBCICodeBase
{
	uint64_t& as_uint64() { return *((uint64_t*)this); }
	uint64_t const& as_uint64() const { return *((uint64_t*)this); }

	template <size_t Bbits>
	PVBCICode<Bbits> const& as() const { return *((PVBCICode<Bbits>*)this); }

	template <size_t Bbits>
	PVBCICode<Bbits>& as() { return *((PVBCICode<Bbits>*)this); }

private:
	uint64_t _int_v;
};

}

#endif
