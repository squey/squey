#include <common/common.h>
#include <code_bz/serial_bcodecb.h>
#include <cassert>

#include <pvkernel/core/picviz_intrin.h>

void serial_bcodecb(PVBCode* codes, size_t n, BCodeCB cb)
{
	for (size_t i = 0; i < n; i++) {
		PVBCode bit = codes[i];
		assert_bcode_valid(bit);
		B_SET(cb[(bit.int_v)>>5], ((bit.int_v)&31));
	}
}

#if 0
void sse_bcodecb(PVBCode* codes, size_t n, BCodeCB cb)
{
	__m128i sse_codes;
	__m128i sse_bits;
	__m128i sse_mask;
	__m128i sse_cb;
	__m128i sse_and31 = _mm_set_epi32(31, 31, 31, 31);
	__m128i sse_1 = _mm_set_epi32(1, 1, 1, 1);
	uint32_t DECLARE_ALIGN(16) idxes[4], bits[4], tmp_cb[4];
	for (size_t i = 0; i < n; i += 4) {
		sse_codes = _mm_load_si128((__m128i*) &codes[i]);
		sse_bits = _mm_and_si128(sse_codes, sse_and31);
		sse_codes = _mm_srli_epi32(sse_codes, 5);
		sse_mask = _mm_sll_epi32(sse_1, sse_bits);

		// Load the collision buffer
		_mm_store_si128((__m128i*) idxes, sse_codes);
		sse_cb = _mm_insert_epi32(sse_cb, cb[idxes[0]], 0);
		sse_cb = _mm_insert_epi32(sse_cb, cb[idxes[1]], 1);
		sse_cb = _mm_insert_epi32(sse_cb, cb[idxes[2]], 2);
		sse_cb = _mm_insert_epi32(sse_cb, cb[idxes[3]], 3);
		sse_cb = _mm_or_si128(sse_cb, sse_mask);
		_mm_store_si128((__m128i*) tmp_cb, sse_cb);
		cb[idxes[0]] = tmp_cb[0];
		cb[idxes[1]] = tmp_cb[1];
		cb[idxes[2]] = tmp_cb[2];
		cb[idxes[3]] = tmp_cb[3];

	}
	for (size_t i = (n/4)*4; i < n; i++) {
		PVBCode bit = codes[i];
		assert_bcode_valid(bit);
		B_SET(cb[(bit.int_v)>>5], ((bit.int_v)&31));
	}
}
#endif
