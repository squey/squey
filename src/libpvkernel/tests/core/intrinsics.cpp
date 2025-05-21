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
#include <pvkernel/core/squey_assert.h>
#include <cstdio>

simde__m128i get_epi8_ff_pos(int pos)
{
	if (pos <= 7) {
		const simde__m128i v = simde_mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xFF);
		return simde_mm_slli_epi64(v, pos * 8);
	}

	const simde__m128i v = simde_mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0);
	return simde_mm_slli_epi64(v, (pos - 8) * 8);
}

void test_pos_lastnonzero_8(simde__m128i const v, int ref)
{
	int pos = squey_mm_getpos_lastnonzero_epi8(v);
	printf("Test with sse=0x%.16llx%.16llx, pos_lastnonzero_8 = %d\n", simde_mm_extract_epi64(v, 1),
	       simde_mm_extract_epi64(v, 0), pos);
	PV_ASSERT_VALID(pos == ref);
}

int main()
{
	const simde__m128i sse_ff = simde_mm_set1_epi32(0xFFFFFFFF);
	// Test squey_mm_getpos_lastff_epi8
	for (int i = 0; i < 16; i++) {
		const simde__m128i sse_v = get_epi8_ff_pos(i);
		printf("Test with 0xff at the %d byte,\tsse=0x%.16llx%.16llx", i,
		       simde_mm_extract_epi64(sse_v, 1), simde_mm_extract_epi64(sse_v, 0));
		int pos_nonzero_8 = squey_mm_getpos_nonzero_epi8(sse_v, sse_ff);
		int pos_max_16 = squey_mm_getpos_max_epi16(sse_v, sse_ff);
		int pos_lastnonzero_8 = squey_mm_getpos_lastnonzero_epi8(sse_v);
		printf(", pos_nonzero_8 = %d, pos_max_16 = %d, pos_lastnonzero_8 = %d\n", pos_nonzero_8,
		       pos_max_16, pos_lastnonzero_8);
		PV_ASSERT_VALID(pos_nonzero_8 == i);
		PV_ASSERT_VALID(pos_max_16 == ((i & ~1) >> 1));
		PV_ASSERT_VALID(pos_lastnonzero_8 == i);
	}

	simde__m128i sse_v =
	    simde_mm_set_epi8(0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0);
	for (int i = 15; i >= 0; i--) {
		test_pos_lastnonzero_8(sse_v, i);
		sse_v = simde_mm_srli_si128(sse_v, 1);
	}

	sse_v = sse_ff;
	for (int i = 15; i >= 0; i--) {
		test_pos_lastnonzero_8(sse_v, i);
		sse_v = simde_mm_srli_si128(sse_v, 1);
	}

	sse_v = simde_mm_set_epi32(-1, 5, 6, 7);
	PV_ASSERT_VALID(squey_mm_hmin_epi32(sse_v) == -1);
	sse_v = simde_mm_set_epi32(5, -1, 6, 7);
	PV_ASSERT_VALID(squey_mm_hmin_epi32(sse_v) == -1);
	sse_v = simde_mm_set_epi32(5, 6, -1, 7);
	PV_ASSERT_VALID(squey_mm_hmin_epi32(sse_v) == -1);
	sse_v = simde_mm_set_epi32(5, 6, 7, -1);
	PV_ASSERT_VALID(squey_mm_hmin_epi32(sse_v) == -1);

	sse_v = simde_mm_set_epi32(0, 5, 6, 7);
	PV_ASSERT_VALID(squey_mm_hmin_epu32(sse_v) == 0);
	sse_v = simde_mm_set_epi32(5, 0, 6, 7);
	PV_ASSERT_VALID(squey_mm_hmin_epu32(sse_v) == 0);
	sse_v = simde_mm_set_epi32(5, 6, 0, 7);
	PV_ASSERT_VALID(squey_mm_hmin_epu32(sse_v) == 0);
	sse_v = simde_mm_set_epi32(5, 6, 7, 0);
	PV_ASSERT_VALID(squey_mm_hmin_epu32(sse_v) == 0);

	return 0;
}
