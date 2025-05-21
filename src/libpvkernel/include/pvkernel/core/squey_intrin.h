/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVKERNEL_CORE_SQUEY_INTRIN_H
#define PVKERNEL_CORE_SQUEY_INTRIN_H

#include <simde/x86/sse4.1.h>
#include <simde/x86/sse2.h>
#include <simde/x86/avx.h>
#include <simde/x86/avx2.h>

#include <cstdint>
#include <stdexcept>

// Pïcviz's helpers

// Get the position of the maximum 16-bits word in v
// sse_ff must be a vector full of 0xFF (must be given for performance reasons)
inline static int squey_mm_getpos_max_epi16(simde__m128i v, simde__m128i const sse_ff)
{
	// Get ~v
	v = simde_mm_andnot_si128(v, sse_ff);

	// Get the position of the minimum 16-bit word
	const simde__m128i sse_min = simde_mm_minpos_epu16(v);

	// The index is in sse_min[18:16] (inclusive)
	return simde_mm_extract_epi8(sse_min, 2) & 7U;
}

// Assumes that v is full of zero but with one 8-bits word equals to non-zero
// This functions finds the position of that 8-bits word.
inline static int squey_mm_getpos_nonzero_epi8(simde__m128i v, simde__m128i const sse_ff)
{
	// Get ~v
	v = simde_mm_andnot_si128(v, sse_ff);

	// Get the position of the minimum 16-bit word
	const simde__m128i sse_min = simde_mm_minpos_epu16(v);

	// The index is in sse_min[18:16] (inclusive)
	const int pos16 = simde_mm_extract_epi8(sse_min, 2) & 7U;

	// and the maximum value in sse_min[0:15]
	return pos16 << 1 | (simde_mm_extract_epi8(sse_min, 1) == 0);
}

// This functions finds the position of the *last* non-null 8-bit word of 'v'
inline static int squey_mm_getpos_lastnonzero_epi8(simde__m128i const ssev)
{
	uint64_t v = simde_mm_extract_epi64(ssev, 1);
	int pos64 = 1;

	if (v == 0) {
		v = simde_mm_extract_epi64(ssev, 0);
		pos64 = 0;
	}

	const int pos32 = ((v & 0xFFFFFFFF00000000ULL) != 0);
	v >>= pos32 << 5; // *32

	const int pos16 = ((v & 0xFFFF0000ULL) != 0);
	v >>= pos16 << 4; // *16

	//           *8         *4         *2
	return pos64 << 3 | pos32 << 2 | pos16 << 1 | ((v & 0xFF00ULL) != 0);
}

inline static int mm_popcnt_u128(simde__m128i const v)
{
	const uint64_t b0 = simde_mm_extract_epi64(v, 0);
	const uint64_t b1 = simde_mm_extract_epi64(v, 1);
	return __builtin_popcountll(b0) + __builtin_popcountll(b1);
}

inline static simde__m256d squey_mm256_cvtepu32_pd(simde__m128i const v)
{
	const simde__m128i mask_carry_sse = simde_mm_set1_epi32(0x7fffffff);
	const simde__m256d conv_31 = simde_mm256_cvtepi32_pd(simde_mm_and_si128(v, mask_carry_sse));
	const simde__m256d v_carry_dble = simde_mm256_cvtepi32_pd(simde_mm_srli_epi32(v, 31));
	return simde_mm256_add_pd(conv_31, simde_mm256_mul_pd(v_carry_dble, simde_mm256_set1_pd(1U << 31)));
}

inline static simde__m128i squey_mm256_cvttpd_epu32(simde__m256d const v)
{
	const simde__m256d avx_31_pd = simde_mm256_set1_pd(1U << 31);
	const simde__m256d mask_over = simde_mm256_cmp_pd(v, avx_31_pd, SIMDE_CMP_GT_OQ);
	const simde__m256d v_under = simde_mm256_sub_pd(v, simde_mm256_and_pd(avx_31_pd, mask_over));
	simde__m128i ret = simde_mm256_cvttpd_epi32(v_under);

	const uint64_t bitmask = simde_mm256_movemask_pd(mask_over);
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
#endif
	simde__m128i mask_sse;
	mask_sse = simde_mm_insert_epi64(mask_sse, ((bitmask & 1U) << 31) | ((bitmask & 2U) << 62), 0);
	mask_sse = simde_mm_insert_epi64(mask_sse, ((bitmask & 4U) << 29) | ((bitmask & 8U) << 60), 1);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
	return simde_mm_or_si128(ret, mask_sse);
}

/*! \brief This intrinsics emulation compare packed unsigned 32-bit signed integers for strict
 *"less-than"
 * The result using simde_mm_cmplt_epi32 (signed version) is wrong iif only one of the two operand is
 *negative.
 * This function fixes that issue.
 *
 * res[i] = (a[i] < b[i])
 */
inline simde__m128i squey_mm_cmplt_epu32(simde__m128i const a, simde__m128i const b)
{
	const simde__m128i cmp_signed = simde_mm_cmplt_epi32(a, b);
	const simde__m128i cmp_a_31 = simde_mm_cmplt_epi32(a, simde_mm_setzero_si128());
	const simde__m128i cmp_b_31 = simde_mm_cmplt_epi32(b, simde_mm_setzero_si128());
	return reinterpret_cast<simde__m128i>(simde_mm_blendv_ps(
	    reinterpret_cast<simde__m128>(cmp_signed),                          // if mask[i][31] == 0
	    reinterpret_cast<simde__m128>(cmp_b_31),                            // if mask[i][31] == 1
	    reinterpret_cast<simde__m128>(simde_mm_xor_si128(cmp_a_31, cmp_b_31)))); // "mask" register
}

/*! \brief This intrinsics emulation compare packed signed 32-bit integers for inclusive within a
 *range [a, b[
 *
 * res[i] = (v[i] >= a[i]) && (v[i] < b[i])
 */
inline simde__m128i squey_mm_cmprange_epi32(simde__m128i const v, simde__m128i const a, simde__m128i const b)
{
	// simde_mm_andnot_si128(a,b) = ~a & b
	// simde_mm_cmplt_epi32(a,b) = a < b;
	// thus andnot(cmplt(a,b),cmplt(a,c)) <=> (!(a < b)) && (a < c) <=> (a >=b) && (a < c)
	return simde_mm_andnot_si128(simde_mm_cmplt_epi32(v, a), simde_mm_cmplt_epi32(v, b));
}

/*! \brief This intrinsics emulation compare packed signed 32-bit integers for inclusive within a
 *range [a, b]
 *
 * res[i] = (v[i] >= a[i]) && (v[i] < b[i])
 */
inline simde__m128i squey_mm_cmprange_in_epi32(simde__m128i const v, simde__m128i const a, simde__m128i const b)
{
	// same logic as previous function
	return simde_mm_andnot_si128(simde_mm_cmplt_epi32(v, a),
	       simde_mm_or_si128(simde_mm_cmplt_epi32(v, b), simde_mm_cmpeq_epi32(v, b)));
}

/*! \brief This intrinsics emulation compare packed unsigned 32-bit integers for inclusive within a
 *range [a, b[
 *
 * res[i] = (v[i] >= a[i]) && (v[i] < b[i])
 */
inline static simde__m128i squey_mm_cmprange_epu32(simde__m128i const v, simde__m128i const a, simde__m128i const b)
{
	return simde_mm_andnot_si128(squey_mm_cmplt_epu32(v, a), squey_mm_cmplt_epu32(v, b));
}

/*! \brief This intrinsics emulation compare packed unsigned 32-bit integers for inclusive within a
 *range [a, b]
 *
 * res[i] = (v[i] >= a[i]) && (v[i] <= b[i])
 */
inline static simde__m128i squey_mm_cmprange_in_epu32(simde__m128i const v, simde__m128i const a, simde__m128i const b)
{
	return simde_mm_andnot_si128(squey_mm_cmplt_epu32(v, a),
	                        simde_mm_or_si128(squey_mm_cmplt_epu32(v, b), simde_mm_cmpeq_epi32(v, b)));
}

/*! \brief Returns the minimum 32-bit signed integer in a packed 32-bit signed integer vector
 *
 * \return min({v[i], i=0..4})
 */
inline static int32_t squey_mm_hmin_epi32(simde__m128i const v)
{
	simde__m128i min_perm = simde_mm_shuffle_epi32(v, (2 | (3 << 2)));
	simde__m128i min = simde_mm_min_epi32(v, min_perm);

	min_perm = simde_mm_shuffle_epi32(min, 1);
	min = simde_mm_min_epi32(min, min_perm);

	return simde_mm_extract_epi32(min, 0);
}

/*! \brief Returns the minimum 32-bit unsigned integer in a packed 32-bit unsigned integer vector
 *
 * \return min({v[i], i=0..4})
 */
inline static uint32_t squey_mm_hmin_epu32(simde__m128i const v)
{
	simde__m128i min_perm = simde_mm_shuffle_epi32(v, (2 | (3 << 2)));
	simde__m128i min = simde_mm_min_epu32(v, min_perm);

	min_perm = simde_mm_shuffle_epi32(min, 1);
	min = simde_mm_min_epu32(min, min_perm);

	return simde_mm_extract_epi32(min, 0);
}

#if defined(__aarch64__) || defined(_M_ARM64)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmacro-redefined"

inline static simde__m128i squey_mm_slli_epi32(simde__m128i value, int count)
{
	switch (count) {
		case 0: return value;
		case 1: return simde_mm_slli_epi32(value, 1);
		case 2: return simde_mm_slli_epi32(value, 2);
		case 3: return simde_mm_slli_epi32(value, 3);
		case 4: return simde_mm_slli_epi32(value, 4);
		case 5: return simde_mm_slli_epi32(value, 5);
		case 6: return simde_mm_slli_epi32(value, 6);
		case 7: return simde_mm_slli_epi32(value, 7);
		case 8: return simde_mm_slli_epi32(value, 8);
		case 9: return simde_mm_slli_epi32(value, 9);
		case 10: return simde_mm_slli_epi32(value, 10);
		case 11: return simde_mm_slli_epi32(value, 11);
		case 12: return simde_mm_slli_epi32(value, 12);
		case 13: return simde_mm_slli_epi32(value, 13);
		case 14: return simde_mm_slli_epi32(value, 14);
		case 15: return simde_mm_slli_epi32(value, 15);
		case 16: return simde_mm_slli_epi32(value, 16);
		case 17: return simde_mm_slli_epi32(value, 17);
		case 18: return simde_mm_slli_epi32(value, 18);
		case 19: return simde_mm_slli_epi32(value, 19);
		case 20: return simde_mm_slli_epi32(value, 20);
		case 21: return simde_mm_slli_epi32(value, 21);
		case 22: return simde_mm_slli_epi32(value, 22);
		case 23: return simde_mm_slli_epi32(value, 23);
		case 24: return simde_mm_slli_epi32(value, 24);
		case 25: return simde_mm_slli_epi32(value, 25);
		case 26: return simde_mm_slli_epi32(value, 26);
		case 27: return simde_mm_slli_epi32(value, 27);
		case 28: return simde_mm_slli_epi32(value, 28);
		case 29: return simde_mm_slli_epi32(value, 29);
		case 30: return simde_mm_slli_epi32(value, 30);
		case 31: return simde_mm_slli_epi32(value, 31);
		default: throw std::invalid_argument("Invalid shift value");
    }
}

inline static simde__m128i squey_mm_srli_epi32(simde__m128i value, int count)
{
	switch (count) {
		case 0: return value;
		case 1: return simde_mm_srli_epi32(value, 1);
		case 2: return simde_mm_srli_epi32(value, 2);
		case 3: return simde_mm_srli_epi32(value, 3);
		case 4: return simde_mm_srli_epi32(value, 4);
		case 5: return simde_mm_srli_epi32(value, 5);
		case 6: return simde_mm_srli_epi32(value, 6);
		case 7: return simde_mm_srli_epi32(value, 7);
		case 8: return simde_mm_srli_epi32(value, 8);
		case 9: return simde_mm_srli_epi32(value, 9);
		case 10: return simde_mm_srli_epi32(value, 10);
		case 11: return simde_mm_srli_epi32(value, 11);
		case 12: return simde_mm_srli_epi32(value, 12);
		case 13: return simde_mm_srli_epi32(value, 13);
		case 14: return simde_mm_srli_epi32(value, 14);
		case 15: return simde_mm_srli_epi32(value, 15);
		case 16: return simde_mm_srli_epi32(value, 16);
		case 17: return simde_mm_srli_epi32(value, 17);
		case 18: return simde_mm_srli_epi32(value, 18);
		case 19: return simde_mm_srli_epi32(value, 19);
		case 20: return simde_mm_srli_epi32(value, 20);
		case 21: return simde_mm_srli_epi32(value, 21);
		case 22: return simde_mm_srli_epi32(value, 22);
		case 23: return simde_mm_srli_epi32(value, 23);
		case 24: return simde_mm_srli_epi32(value, 24);
		case 25: return simde_mm_srli_epi32(value, 25);
		case 26: return simde_mm_srli_epi32(value, 26);
		case 27: return simde_mm_srli_epi32(value, 27);
		case 28: return simde_mm_srli_epi32(value, 28);
		case 29: return simde_mm_srli_epi32(value, 29);
		case 30: return simde_mm_srli_epi32(value, 30);
		case 31: return simde_mm_srli_epi32(value, 31);
		default: throw std::invalid_argument("Invalid shift value");
    }
}

inline static simde__m128i squey_mm_slli_epi64(simde__m128i value, int count)
{
	switch (count) {
		case 0: return value;
		case 1: return simde_mm_slli_epi64(value, 1);
		case 2: return simde_mm_slli_epi64(value, 2);
		case 3: return simde_mm_slli_epi64(value, 3);
		case 4: return simde_mm_slli_epi64(value, 4);
		case 5: return simde_mm_slli_epi64(value, 5);
		case 6: return simde_mm_slli_epi64(value, 6);
		case 7: return simde_mm_slli_epi64(value, 7);
		case 8: return simde_mm_slli_epi64(value, 8);
		case 9: return simde_mm_slli_epi64(value, 9);
		case 10: return simde_mm_slli_epi64(value, 10);
		case 11: return simde_mm_slli_epi64(value, 11);
		case 12: return simde_mm_slli_epi64(value, 12);
		case 13: return simde_mm_slli_epi64(value, 13);
		case 14: return simde_mm_slli_epi64(value, 14);
		case 15: return simde_mm_slli_epi64(value, 15);
		case 16: return simde_mm_slli_epi64(value, 16);
		case 17: return simde_mm_slli_epi64(value, 17);
		case 18: return simde_mm_slli_epi64(value, 18);
		case 19: return simde_mm_slli_epi64(value, 19);
		case 20: return simde_mm_slli_epi64(value, 20);
		case 21: return simde_mm_slli_epi64(value, 21);
		case 22: return simde_mm_slli_epi64(value, 22);
		case 23: return simde_mm_slli_epi64(value, 23);
		case 24: return simde_mm_slli_epi64(value, 24);
		case 25: return simde_mm_slli_epi64(value, 25);
		case 26: return simde_mm_slli_epi64(value, 26);
		case 27: return simde_mm_slli_epi64(value, 27);
		case 28: return simde_mm_slli_epi64(value, 28);
		case 29: return simde_mm_slli_epi64(value, 29);
		case 30: return simde_mm_slli_epi64(value, 30);
		case 31: return simde_mm_slli_epi64(value, 31);
		case 32: return simde_mm_slli_epi64(value, 32);
		case 33: return simde_mm_slli_epi64(value, 33);
		case 34: return simde_mm_slli_epi64(value, 34);
		case 35: return simde_mm_slli_epi64(value, 35);
		case 36: return simde_mm_slli_epi64(value, 36);
		case 37: return simde_mm_slli_epi64(value, 37);
		case 38: return simde_mm_slli_epi64(value, 38);
		case 39: return simde_mm_slli_epi64(value, 39);
		case 40: return simde_mm_slli_epi64(value, 40);
		case 41: return simde_mm_slli_epi64(value, 41);
		case 42: return simde_mm_slli_epi64(value, 42);
		case 43: return simde_mm_slli_epi64(value, 43);
		case 44: return simde_mm_slli_epi64(value, 44);
		case 45: return simde_mm_slli_epi64(value, 45);
		case 46: return simde_mm_slli_epi64(value, 46);
		case 47: return simde_mm_slli_epi64(value, 47);
		case 48: return simde_mm_slli_epi64(value, 48);
		case 49: return simde_mm_slli_epi64(value, 49);
		case 50: return simde_mm_slli_epi64(value, 50);
		case 51: return simde_mm_slli_epi64(value, 51);
		case 52: return simde_mm_slli_epi64(value, 52);
		case 53: return simde_mm_slli_epi64(value, 53);
		case 54: return simde_mm_slli_epi64(value, 54);
		case 55: return simde_mm_slli_epi64(value, 55);
		case 56: return simde_mm_slli_epi64(value, 56);
		case 57: return simde_mm_slli_epi64(value, 57);
		case 58: return simde_mm_slli_epi64(value, 58);
		case 59: return simde_mm_slli_epi64(value, 59);
		case 60: return simde_mm_slli_epi64(value, 60);
		case 61: return simde_mm_slli_epi64(value, 61);
		case 62: return simde_mm_slli_epi64(value, 62);
		case 63: return simde_mm_slli_epi64(value, 63);
		default: throw std::invalid_argument("Invalid shift value");
    }
}

#define simde_mm_srli_epi32 squey_mm_srli_epi32
#define simde_mm_slli_epi32 squey_mm_slli_epi32
#define simde_mm_slli_epi64 squey_mm_slli_epi64
#pragma GCC diagnostic pop
#endif

#endif
