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

#ifndef PVPARALLELVIEW_PVHITGRAPHSSEHELPERS_H
#define PVPARALLELVIEW_PVHITGRAPHSSEHELPERS_H

#ifdef __AVX__
#define HCSSE_ALPHA_DOUBLE_VEC __m256d
#else
#define HCSSE_ALPHA_DOUBLE_VEC __m128d
#endif

namespace PVParallelView
{

struct PVHitGraphSSEHelpers {
	static __m128i buffer_offset_from_y_sse(__m128i y_sse,
	                                        __m128i p_sse,
	                                        const __m128i y_min_ref_sse,
	                                        const HCSSE_ALPHA_DOUBLE_VEC alpha_sse,
	                                        const __m128i zoom_mask_sse,
	                                        uint32_t idx_shift,
	                                        uint32_t zoom_shift,
	                                        size_t nbits);
};
} // namespace PVParallelView

#endif
