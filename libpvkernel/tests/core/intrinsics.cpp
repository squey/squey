#include <pvkernel/core/picviz_intrin.h>
#include <pvkernel/core/picviz_assert.h>
#include <stdio.h>

__m128i get_epi8_ff_pos(int pos)
{
	if (pos <= 7) {
		const __m128i v = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0,
									   0, 0, 0, 0, 0, 0, 0, 0xFF);
		return _mm_slli_epi64(v, pos*8);
	}

	const __m128i v = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0xFF,
								   0, 0, 0, 0, 0, 0, 0, 0);
	return _mm_slli_epi64(v, (pos-8)*8);
}

void test_pos_lastnonzero_8(__m128i const v, int ref)
{
	int pos = picviz_mm_getpos_lastnonzero_epi8(v);
	printf("Test with sse=0x%.16llx%.16llx, pos_lastnonzero_8 = %d\n", _mm_extract_epi64(v, 1), _mm_extract_epi64(v, 0), pos);
	PV_ASSERT_VALID(pos == ref);
}

int main()
{
	const __m128i sse_ff = _mm_set1_epi32(0xFFFFFFFF);
	// Test picviz_mm_getpos_lastff_epi8
	for (int i = 0; i < 16; i++) {
		const __m128i sse_v = get_epi8_ff_pos(i);
		printf("Test with 0xff at the %d byte,\tsse=0x%.16llx%.16llx", i, _mm_extract_epi64(sse_v, 1), _mm_extract_epi64(sse_v, 0));
		int pos_nonzero_8 = picviz_mm_getpos_nonzero_epi8(sse_v, sse_ff);
		int pos_max_16 = picviz_mm_getpos_max_epi16(sse_v, sse_ff);
		int pos_lastnonzero_8 = picviz_mm_getpos_lastnonzero_epi8(sse_v);
		printf(", pos_nonzero_8 = %d, pos_max_16 = %d, pos_lastnonzero_8 = %d\n", pos_nonzero_8, pos_max_16, pos_lastnonzero_8);
		PV_ASSERT_VALID(pos_nonzero_8 == i);
		PV_ASSERT_VALID(pos_max_16 == ((i&~1)>>1));
		PV_ASSERT_VALID(pos_lastnonzero_8 == i);
	}

	__m128i sse_v = _mm_set_epi8(0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0,
	                             0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0);
	for (int i = 15; i >= 0; i--) {
		test_pos_lastnonzero_8(sse_v, i);
		sse_v = _mm_srli_si128(sse_v, 1);
	}

	sse_v = sse_ff;
	for (int i = 15; i >= 0; i--) {
		test_pos_lastnonzero_8(sse_v, i);
		sse_v = _mm_srli_si128(sse_v, 1);
	}

	return 0;
}
