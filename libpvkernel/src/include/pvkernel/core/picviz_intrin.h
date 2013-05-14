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
inline static int picviz_mm_getpos_max_epi16(__m128i v, __m128i const sse_ff)
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
inline static int picviz_mm_getpos_nonzero_epi8(__m128i v, __m128i const sse_ff)
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
inline static int picviz_mm_getpos_lastnonzero_epi8(__m128i const ssev)
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

inline static int mm_popcnt_u128(__m128i const v)
{
	const uint64_t b0 = _mm_extract_epi64(v, 0);
	const uint64_t b1 = _mm_extract_epi64(v, 1); 
	return _mm_popcnt_u64(b0) + _mm_popcnt_u64(b1);
}

#ifdef __AVX__
inline static __m256d picviz_mm256_cvtepu32_pd(__m128i const v)
{
	const __m128i mask_carry_sse = _mm_set1_epi32(0x7fffffff);
	const __m256d conv_31 = _mm256_cvtepi32_pd(_mm_and_si128(v, mask_carry_sse));
	const __m256d v_carry_dble = _mm256_cvtepi32_pd(_mm_srli_epi32(v, 31));
	return _mm256_add_pd(conv_31, _mm256_mul_pd(v_carry_dble, _mm256_set1_pd(1U<<31)));
}

inline static __m128i picviz_mm256_cvttpd_epu32(__m256d const v)
{
	const __m256d avx_31_pd = _mm256_set1_pd(1U<<31);
	const __m256d mask_over = _mm256_cmp_pd(v, avx_31_pd, _CMP_GT_OQ);
	const __m256d v_under = _mm256_sub_pd(v, _mm256_and_pd(avx_31_pd, mask_over));
	__m128i ret = _mm256_cvttpd_epi32(v_under);

	const uint64_t bitmask = _mm256_movemask_pd(mask_over);
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
#endif
	__m128i mask_sse;
	mask_sse = _mm_insert_epi64(mask_sse, ((bitmask & 1U)<<31) | ((bitmask & 2U) << 62), 0); 
	mask_sse = _mm_insert_epi64(mask_sse, ((bitmask & 4U)<<29) | ((bitmask & 8U) << 60), 1); 
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
	return _mm_or_si128(ret, mask_sse);
}
#endif

#ifdef __SSE2__
inline static __m128d picviz_mm_cvtepu32_pd(__m128i const v)
{
	const __m128i mask_carry_sse = _mm_set1_epi32(0x7fffffff);
	__m128d conv_31 = _mm_cvtepi32_pd(_mm_and_si128(v, mask_carry_sse));
	__m128i v_carry = _mm_srli_epi32(v, 31);
	__m128d v_carry_dble = _mm_cvtepi32_pd(v_carry);
	return _mm_add_pd(conv_31, _mm_mul_pd(v_carry_dble, _mm_set1_pd(1U<<31)));
}

inline static __m128i picviz_mm_cvttpd_epu32(__m128d const v)
{
	const __m128d sse_31_pd = _mm_set1_pd(1U<<31);
	const __m128d mask_over = _mm_cmpgt_pd(v, sse_31_pd);
	__m128d v_under = _mm_sub_pd(v, _mm_and_pd(sse_31_pd, mask_over));
	__m128i ret = _mm_cvttpd_epi32(v_under);

	__m128i mask_over_int = reinterpret_cast<__m128i>(_mm_shuffle_epi32(reinterpret_cast<__m128i>(mask_over), 0 | (2 << 2) | (1 << 4) | (3 << 6)));

	return _mm_or_si128(ret, _mm_and_si128(mask_over_int, _mm_set1_epi32(1U<<31)));
}
#endif

/*! \brief This intrinsics emulation compare packed unsigned 32-bit signed integers for strict "less-than"
 * The result using _mm_cmplt_epi32 (signed version) is wrong iif only one of the two operand is negative.
 * This function fixes that issue.
 *
 * res[i] = (a[i] < b[i])
 */
inline __m128i picviz_mm_cmplt_epu32(__m128i const a, __m128i const b)
{
	const __m128i cmp_signed = _mm_cmplt_epi32(a, b);
	const __m128i cmp_a_31 = _mm_cmplt_epi32(a, _mm_setzero_si128());
	const __m128i cmp_b_31 = _mm_cmplt_epi32(b, _mm_setzero_si128());
	return reinterpret_cast<__m128i>(_mm_blendv_ps(reinterpret_cast<__m128>(cmp_signed), // if mask[i][31] == 0
	                                               reinterpret_cast<__m128>(cmp_b_31),   // if mask[i][31] == 1
	                                               reinterpret_cast<__m128>(_mm_xor_si128(cmp_a_31, cmp_b_31)))); // "mask" register
}

/*! \brief This intrinsics emulation compare packed signed 32-bit integers for inclusive within a range [a, b[
 *
 * res[i] = (v[i] >= a[i]) && (v[i] < b[i])
 */
inline __m128i picviz_mm_cmprange_epi32(__m128i const v, __m128i const a, __m128i const b)
{
	// _mm_andnot_si128(a,b) = ~a & b
	// _mm_cmplt_epi32(a,b) = a < b;
	// thus andnot(cmplt(a,b),cmplt(a,c)) <=> (!(a < b)) && (a < c) <=> (a >=b) && (a < c)
	return _mm_andnot_si128(_mm_cmplt_epi32(v, a),
	                        _mm_cmplt_epi32(v, b));
}

/*! \brief This intrinsics emulation compare packed unsigned 32-bit integers for inclusive within a range [a, b[
 *
 * res[i] = (v[i] >= a[i]) && (v[i] < b[i])
 */
inline __m128i picviz_mm_cmprange_epu32(__m128i const v, __m128i const a, __m128i const b)
{
	return _mm_andnot_si128(picviz_mm_cmplt_epu32(v, a),
	                        picviz_mm_cmplt_epu32(v, b));
}

/*! \brief This intrinsics emulation compare packed unsigned 32-bit integers for inclusive within a range [a, b]
 *
 * res[i] = (v[i] >= a[i]) && (v[i] <= b[i])
 */
inline __m128i picviz_mm_cmprange_in_epu32(__m128i const v, __m128i const a, __m128i const b)
{
	return _mm_andnot_si128(picviz_mm_cmplt_epu32(v, a),
	                        _mm_or_si128(picviz_mm_cmplt_epu32(v, b),
	                                     _mm_cmpeq_epi32(v, b)));
}

#endif
