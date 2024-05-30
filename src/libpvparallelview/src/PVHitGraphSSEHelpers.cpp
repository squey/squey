//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/core/squey_intrin.h>
#include <pvparallelview/PVHitGraphSSEHelpers.h>

__m128i PVParallelView::PVHitGraphSSEHelpers::buffer_offset_from_y_sse(
    __m128i y_sse,
    __m128i p_sse,
    const __m128i y_min_ref_sse,
    const HCSSE_ALPHA_DOUBLE_VEC alpha_sse,
    const __m128i zoom_mask_sse,
    uint32_t idx_shift,
    uint32_t zoom_shift,
    size_t nbits)
{
/* y = (y - y_min_ref) * alpha
 * p = y >> zoom_shift (i.e. base)
 * off =  = (p << nbits) + (y & zoom_mask) >> idx_shift
 */
#ifdef __AVX__
	y_sse = _mm_sub_epi32(y_sse, y_min_ref_sse);
	const __m256d tmp1_avx = squey_mm256_cvtepu32_pd(y_sse);
	const __m256d tmp2_avx = _mm256_mul_pd(tmp1_avx, alpha_sse);
	y_sse = squey_mm256_cvttpd_epu32(tmp2_avx);

#elif defined __SSE2__
	y_sse = _mm_sub_epi32(y_sse, y_min_ref_sse);
	const __m128i tmp1_lo_sse = _mm_shuffle_epi32(y_sse, _MM_SHUFFLE(0, 0, 2, 0));
	const __m128i tmp1_hi_sse = _mm_shuffle_epi32(y_sse, _MM_SHUFFLE(0, 0, 3, 1));
	const __m128d tmp2_lo_sse = squey_mm_cvtepu32_pd(tmp1_lo_sse);
	const __m128d tmp2_hi_sse = squey_mm_cvtepu32_pd(tmp1_hi_sse);
	const __m128d tmp3_lo_sse = _mm_mul_pd(tmp2_lo_sse, alpha_sse);
	const __m128d tmp3_hi_sse = _mm_mul_pd(tmp2_hi_sse, alpha_sse);
	const __m128i tmp4_lo_sse = squey_mm_cvttpd_epu32(tmp3_lo_sse);
	const __m128i tmp4_hi_sse = squey_mm_cvttpd_epu32(tmp3_hi_sse);

	y_sse = _mm_unpacklo_epi32(tmp4_lo_sse, tmp4_hi_sse);

#else
#error you need at least SSE2 intrinsics
#endif

	p_sse = _mm_srli_epi32(y_sse, zoom_shift);
	const __m128i off_sse =
	    _mm_add_epi32(_mm_slli_epi32(p_sse, nbits),
	                  _mm_srli_epi32(_mm_and_si128(y_sse, zoom_mask_sse), idx_shift));

	return off_sse;
}
