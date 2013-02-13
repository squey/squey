#include <pvkernel/core/PVByteVisitor.h>
#include <pvkernel/core/picviz_intrin.h>

#include <assert.h>

uint8_t const* PVCore::PVByteVisitor::__impl::get_nth_slice_serial(uint8_t const* buffer, size_t sbuf, size_t n, size_t& size_ret)
{
	n++;
	// n now is the number of '\0' that we have to look for
	size_t nfound = 0;
	size_t off_start = 0;
	for (size_t i = 0; i < sbuf; i++) {
		if (buffer[i] == 0) {
			nfound++;
			if (nfound == n) {
				size_ret = i-off_start;
				return &buffer[off_start];
			}
			off_start = i+1;
		}
	}

	return nullptr;
}

uint8_t const* PVCore::PVByteVisitor::__impl::get_nth_slice_sse(uint8_t const* buffer, size_t sbuf, size_t n, size_t& size_ret)
{
	n++;
	// `n' now is the number of '\0' that we have to look for
	size_t nfound = 0;
	const size_t sbuf_sse = sbuf & ~15ULL;

	const __m128i sse_ff = _mm_set1_epi32(0xFFFFFFFF);

	assert(((uintptr_t)buffer % 16) == 0);

	size_t off_start = 0;

	size_t i;
	for (i = 0; i < sbuf_sse; i += 16) {
		const __m128i sse_buf = _mm_load_si128((__m128i const*) &buffer[i]);
		__m128i sse_cmp = _mm_cmpeq_epi8(sse_buf, _mm_setzero_si128());
		if (_mm_test_all_zeros(sse_cmp, sse_ff)) {
			continue;
		}

		const uint64_t b0 = _mm_extract_epi64(sse_cmp, 0);
		const uint64_t b1 = _mm_extract_epi64(sse_cmp, 1); 
		const size_t sse_found = (_mm_popcnt_u64(b0) + _mm_popcnt_u64(b1)) >> 3;
		if ((sse_found == 1) && (nfound + 1) == n) {
			// This branch should be fast as it covers most cases
			size_ret = i + picviz_mm_getpos_nonzero_epi8(sse_cmp, sse_ff) - off_start;
			return &buffer[off_start];
		}

		const size_t new_nfound = nfound + sse_found;
		// Now, sse_found is > 1
		if (new_nfound >= n) {
			// Extract everyone "by hand" until we reach our limit
			if (_mm_extract_epi8(sse_cmp, 0)) {
				nfound++;
				if (nfound == n) {
					size_ret = i + 0 - off_start;
					return &buffer[off_start];
				}
				off_start = i + 0 + 1;
			}
			if (_mm_extract_epi8(sse_cmp, 1)) {
				nfound++;
				if (nfound == n) {
					size_ret = i + 1 - off_start;
					return &buffer[off_start];
				}
				off_start = i + 1 + 1;
			}
			if (_mm_extract_epi8(sse_cmp, 2)) {
				nfound++;
				if (nfound == n) {
					size_ret = i + 2 - off_start;
					return &buffer[off_start];
				}
				off_start = i + 2 + 1;
			}
			if (_mm_extract_epi8(sse_cmp, 3)) {
				nfound++;
				if (nfound == n) {
					size_ret = i + 3 - off_start;
					return &buffer[off_start];
				}
				off_start = i + 3 + 1;
			}
			if (_mm_extract_epi8(sse_cmp, 4)) {
				nfound++;
				if (nfound == n) {
					size_ret = i + 4 - off_start;
					return &buffer[off_start];
				}
				off_start = i + 4 + 1;
			}
			if (_mm_extract_epi8(sse_cmp, 5)) {
				nfound++;
				if (nfound == n) {
					size_ret = i + 5 - off_start;
					return &buffer[off_start];
				}
				off_start = i + 5 + 1;
			}
			if (_mm_extract_epi8(sse_cmp, 6)) {
				nfound++;
				if (nfound == n) {
					size_ret = i + 6 - off_start;
					return &buffer[off_start];
				}
				off_start = i + 6 + 1;
			}
			if (_mm_extract_epi8(sse_cmp, 7)) {
				nfound++;
				if (nfound == n) {
					size_ret = i + 7 - off_start;
					return &buffer[off_start];
				}
				off_start = i + 7 + 1;
			}
			if (_mm_extract_epi8(sse_cmp, 8)) {
				nfound++;
				if (nfound == n) {
					size_ret = i + 8 - off_start;
					return &buffer[off_start];
				}
				off_start = i + 8 + 1;
			}
			if (_mm_extract_epi8(sse_cmp, 9)) {
				nfound++;
				if (nfound == n) {
					size_ret = i + 9 - off_start;
					return &buffer[off_start];
				}
				off_start = i + 9 + 1;
			}
			if (_mm_extract_epi8(sse_cmp, 10)) {
				nfound++;
				if (nfound == n) {
					size_ret = i + 10 - off_start;
					return &buffer[off_start];
				}
				off_start = i + 10 + 1;
			}
			if (_mm_extract_epi8(sse_cmp, 11)) {
				nfound++;
				if (nfound == n) {
					size_ret = i + 11 - off_start;
					return &buffer[off_start];
				}
				off_start = i + 11 + 1;
			}
			if (_mm_extract_epi8(sse_cmp, 12)) {
				nfound++;
				if (nfound == n) {
					size_ret = i + 12 - off_start;
					return &buffer[off_start];
				}
				off_start = i + 12 + 1;
			}
			if (_mm_extract_epi8(sse_cmp, 13)) {
				nfound++;
				if (nfound == n) {
					size_ret = i + 13 - off_start;
					return &buffer[off_start];
				}
				off_start = i + 13 + 1;
			}
			if (_mm_extract_epi8(sse_cmp, 14)) {
				nfound++;
				if (nfound == n) {
					size_ret = i + 14 - off_start;
					return &buffer[off_start];
				}
				off_start = i + 14 + 1;
			}
			if (_mm_extract_epi8(sse_cmp, 15)) {
				nfound++;
				if (nfound == n) {
					size_ret = i + 15 - off_start;
					return &buffer[off_start];
				}
				off_start = i + 15 + 1;
			}
			assert(false);
		}

		nfound = new_nfound;
		off_start = i + picviz_mm_getpos_lastnonzero_epi8(sse_cmp) + 1;
	}
	for (; i < sbuf; i++) {
		if (buffer[i] == 0) {
			nfound++;
			if (nfound == n) {
				size_ret = i-off_start;
				return &buffer[off_start];
			}
			off_start = i+1;
		}
	}

	return nullptr;
}
