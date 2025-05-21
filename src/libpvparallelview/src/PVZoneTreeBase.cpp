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

#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVZoneTree.h>

#include <cassert>

PVParallelView::PVZoneTreeBase::PVZoneTreeBase()
{
	memset(_first_elts, PVROW_INVALID_VALUE, sizeof(PVRow) * NBUCKETS);
	memset(_sel_elts, PVROW_INVALID_VALUE, sizeof(PVRow) * NBUCKETS);
}

size_t PVParallelView::PVZoneTreeBase::browse_tree_bci(PVCore::PVHSVColor const* colors,
                                                       PVBCICode<NBITS_INDEX>* codes) const
{
	return browse_tree_bci_from_buffer(_bg_elts, colors, codes);
}

size_t PVParallelView::PVZoneTreeBase::browse_tree_bci_sel(PVCore::PVHSVColor const* colors,
                                                           PVBCICode<NBITS_INDEX>* codes) const
{
	return browse_tree_bci_from_buffer(_sel_elts, colors, codes);
}

size_t PVParallelView::PVZoneTreeBase::browse_tree_bci_from_buffer(
    const PVRow* elts, PVCore::PVHSVColor const* colors, PVBCICode<NBITS_INDEX>* codes) const
{
	size_t idx_code = 0;

	for (uint64_t b = 0; b < NBUCKETS; b += 4) {

		simde__m128i sse_ff = simde_mm_set1_epi32(0xFFFFFFFF);
		simde__m128i sse_index = simde_mm_load_si128((const simde__m128i*)&elts[b]);
		simde__m128i see_cmp = simde_mm_cmpeq_epi32(sse_ff, sse_index);

		if (simde_mm_testz_si128(see_cmp, sse_ff)) {

			simde__m128i sse_lr = simde_mm_set_epi32(b, b + 1, b + 2, b + 3);

			//  +------------+------------++------------+------------+
			//  |        lr3 |        lr2 ||        lr1 |        lr0 | (sse_lr)
			//  +------------+------------++------------+------------+

			simde__m128i sse_color = simde_mm_set_epi32(colors[simde_mm_extract_epi32(sse_index, 0)].h(),
			                                  colors[simde_mm_extract_epi32(sse_index, 1)].h(),
			                                  colors[simde_mm_extract_epi32(sse_index, 2)].h(),
			                                  colors[simde_mm_extract_epi32(sse_index, 3)].h());
			;
			sse_color = simde_mm_slli_epi32(sse_color, NBITS_INDEX * 2);

			//  +------------+------------++------------+------------+
			//  |color3 << 20|color2 << 20||color1 << 20|color0 << 20| (sse_color)
			//  +------------+------------++------------+------------+

			simde__m128i sse_lrcolor = simde_mm_or_si128(sse_color, sse_lr);

			//  +------------+------------++------------+------------+
			//  |   lrcolor3 |   lrcolor2 ||   lrcolor1 |   lrcolor0 | (sse_lrcolor)
			//  +------------+------------++------------+------------+

			simde__m128i sse_bcicodes0_1 = simde_mm_unpacklo_epi32(sse_index, sse_lrcolor);
			simde__m128i sse_bcicodes2_3 = simde_mm_unpackhi_epi32(sse_index, sse_lrcolor);

			//  +------------+------------++------------+------------+
			//  |   lrcolor1 | index (r1) ||   lrcolor0 | index (r0) |
			//  (sse_bcicodes0_1)
			//  +------------+------------++------------+------------+
			//  +------------+------------++------------+------------+
			//  |   lrcolor3 | index (r3) ||   lrcolor2 | index (r2) |
			//  (sse_bcicodes2_3)
			//  +------------+------------++------------+------------+

			if ((idx_code & 1) == 0) {
				simde_mm_stream_si128((simde__m128i*)&codes[idx_code + 0], sse_bcicodes0_1);
				simde_mm_stream_si128((simde__m128i*)&codes[idx_code + 2], sse_bcicodes2_3);
			} else {
				simde_mm_storeu_si128((simde__m128i*)&codes[idx_code + 0], sse_bcicodes0_1);
				simde_mm_storeu_si128((simde__m128i*)&codes[idx_code + 2], sse_bcicodes2_3);
			}

			idx_code += 4;
		} else {
			for (int i = 0; i < 4; i++) {
				uint64_t b0 = b + i;
				PVRow r = elts[b0];
				if (r != PVROW_INVALID_VALUE) {
					PVBCICode<NBITS_INDEX> bci;
					bci.int_v = r | (b0 << 32);
					bci.s.color = colors[r].h();
					codes[idx_code] = bci;
					idx_code++;
				}
			}
		}
	}

	return idx_code;
}
