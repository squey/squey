#ifndef PVPARALLELVIEW_PVHITGRAPHSSEHELPERS_H
#define PVPARALLELVIEW_PVHITGRAPHSSEHELPERS_H

#ifdef __AVX__
#define HCSSE_ALPHA_DOUBLE_VEC __m256d
#else
#define HCSSE_ALPHA_DOUBLE_VEC __m128d
#endif

namespace PVParallelView {

struct PVHitGraphSSEHelpers
{
	static __m128i buffer_offset_from_y_sse(__m128i y_sse, __m128i p_sse, const __m128i y_min_ref_sse, const HCSSE_ALPHA_DOUBLE_VEC alpha_sse,
                                            const __m128i zoom_mask_sse, uint32_t idx_shift, uint32_t zoom_shift, size_t nbits);
};

}

#endif
