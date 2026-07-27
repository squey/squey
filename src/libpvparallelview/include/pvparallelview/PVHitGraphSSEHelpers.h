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

namespace PVParallelView
{

struct PVHitGraphSSEHelpers {
	/**
	 * @param alpha is taken as a scalar and broadcast inside on purpose. Passing the
	 * 256-bit vector instead makes the caller stage it in its outgoing argument area
	 * with vmovapd, which needs a 32-byte aligned stack. On Windows the functions GCC
	 * outlines from "#pragma omp parallel" -- and every caller here is inside such a
	 * region -- keep only the 16 bytes the ABI guarantees, so that store faults.
	 */
	static simde__m128i buffer_offset_from_y_sse(simde__m128i y_sse,
	                                        simde__m128i p_sse,
	                                        const simde__m128i y_min_ref_sse,
	                                        double alpha,
	                                        const simde__m128i zoom_mask_sse,
	                                        uint32_t idx_shift,
	                                        uint32_t zoom_shift,
	                                        size_t nbits);
};
} // namespace PVParallelView

#endif
