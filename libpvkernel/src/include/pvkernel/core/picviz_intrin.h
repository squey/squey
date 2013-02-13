/**
 * \file picviz_intrin.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVKERNEL_CORE_PICVIZ_INTRIN_H
#define PVKERNEL_CORE_PICVIZ_INTRIN_H

#ifdef WIN32
#define __SSE4_1__
#include <smmintrin.h>
#else
#include <immintrin.h>
#endif

#include <pvkernel/core/general.h>

namespace PVCore {

class LibKernelDecl PVIntrinsics
{
public:
	static bool has_sse41();
	static bool has_sse42();
	static void init_cpuid();

private:
	static bool _has_sse41;
	static bool _has_sse42;
	static bool _init_done;
};

}

// Pïcviz's helpers

// Get the position of the maximum 16-bits word in v
// sse_ff must be a vector full of 0xFF (must be given for performance reasons)
inline int picviz_mm_getpos_max_epi16(__m128i v, __m128i const sse_ff)
{
	// Get ~v
	v = _mm_andnot_si128(v, sse_ff);

	// Get the position of the minimum 16-bit word
	const __m128i sse_min = _mm_minpos_epu16(v);

	// The index is in sse_min[18:16] (inclusive)
	return _mm_extract_epi8(sse_min, 2) & 7U; 
}

// Assumes that v is full of zero but with one 8-bits word equals to non-zero
// This functions finds the position of that 8-bits word.
inline int picviz_mm_getpos_nonzero_epi8(__m128i v, __m128i const sse_ff)
{
	// Get ~v
	v = _mm_andnot_si128(v, sse_ff);

	// Get the position of the minimum 16-bit word
	const __m128i sse_min = _mm_minpos_epu16(v);

	// The index is in sse_min[18:16] (inclusive)
	const int pos16 = _mm_extract_epi8(sse_min, 2) & 7U; 

	// and the maximum value in sse_min[0:15]
	return pos16 << 1 | (_mm_extract_epi8(sse_min, 1) == 0); 
}

// This functions finds the position of the *last* non-null 8-bit word of 'v'
inline int picviz_mm_getpos_lastnonzero_epi8(__m128i const ssev)
{
	uint64_t v = _mm_extract_epi64(ssev, 1);
	int pos64 = 1;

	if (v == 0) {
		v = _mm_extract_epi64(ssev, 0);
		pos64 = 0;
	}

	const int pos32 = ((v & 0xFFFFFFFF00000000ULL) != 0);
	v >>= pos32<<5; // *32

	const int pos16 = ((v & 0xFFFF0000ULL) != 0);
	v >>= pos16<<4; // *16

	//           *8         *4         *2
	return pos64<<3 | pos32<<2 | pos16<<1 | ((v & 0xFF00ULL) != 0);
}

inline int mm_popcnt_u128(__m128i const v)
{
	const uint64_t b0 = _mm_extract_epi64(v, 0);
	const uint64_t b1 = _mm_extract_epi64(v, 1); 
	return _mm_popcnt_u64(b0) + _mm_popcnt_u64(b1);
}



#endif
