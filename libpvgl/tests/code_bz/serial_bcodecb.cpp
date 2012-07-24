/**
 * \file serial_bcodecb.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <common/common.h>
#include <code_bz/serial_bcodecb.h>
#include <cassert>

#include <tbb/parallel_sort.h>
#include <pvkernel/core/picviz_intrin.h>

#include <atomic_ops.h>

#include <omp.h>
#ifdef RED_STATS
#include <map>
#endif

#define COUNT_BITS_UINT32(v,res) \
	v = v - ((v >> 1) & 0x55555555);\
	v = (v & 0x33333333) + ((v >> 2) & 0x33333333);\
	res += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;

static inline uint32_t count_bits(size_t n, const uint32_t* DECLARE_ALIGN(16) data)
{
	uint32_t ret = 0;
	for (size_t i = 0; i < n; i++) {
		uint32_t v = data[i];
		COUNT_BITS_UINT32(v,ret);
	}
	return ret;
}

void serial_bcodecb(PVBCode* codes, size_t n, BCodeCB cb)
{
#ifdef RED_STATS
	std::map<size_t, uint32_t> stats;
#endif
	for (size_t i = 0; i < n; i++) {
		const PVBCode bit = codes[i];
		B_SET(cb[bcode2cb_idx(bit)], bcode2cb_bitn(bit));
#ifdef RED_STATS
		if (i % 10240 == 0) {
			stats[i] = count_bits(NB_INT_BCODECB, cb);
		}
#endif
	}

#ifdef RED_STATS
	FILE* f = fdopen(4, "w");
	std::map<size_t, uint32_t>::iterator it;
	for (it = stats.begin(); it != stats.end(); it++) {
		fprintf(f, "%llu,%u\n", it->first, it->second);
	}
	fprintf(f, "\n");
	fclose(f);
#endif
}

void bcodecb_branch(PVBCode* codes, size_t n, BCodeCB cb)
{
	for (size_t i = 0; i < n; i++) {
		const PVBCode bit = codes[i];
		uint32_t* pcb = &cb[bcode2cb_idx(bit)];
		const uint32_t mask = 1<<(bcode2cb_bitn(bit));
		const uint32_t v = *pcb;
		if ((v & mask) == 0) {
			*pcb = v | mask;
		}
	}
}

void bcodecb_tile(PVBCode* codes, size_t n, BCodeCB cb, uint32_t** tiles_cb)
{
	uint32_t* DECLARE_ALIGN(16) tile_cb;
	for (uint32_t t = 0; t < NB_INT_BCODECB; t += TILE_SIZE_INT) {
		tile_cb = tiles_cb[t/TILE_SIZE_INT];
		for (size_t i = 0; i < n; i++) {
			PVBCode bit = codes[i];
			uint32_t idx = bcode2cb_idx(bit);
			if ((idx >= t) && (idx < t + TILE_SIZE_INT)) {
				B_SET(tile_cb[idx-t], bcode2cb_bitn(bit));
			}
		}
		memcpy(&cb[t], tile_cb, TILE_SIZE_INT*sizeof(uint32_t));
	}
}

void omp_bcodecb_atomic(PVBCode* codes, size_t n, BCodeCB cb)
{
#pragma omp parallel for
	for (size_t i = 0; i < n; i++) {
		const PVBCode bit = codes[i];
		//assert_bcode_valid(bit);
		const uint32_t idx = bcode2cb_idx(bit);
		const uint32_t nbit = bcode2cb_bitn(bit);

#pragma omp atomic
		cb[idx] |= 1<<nbit;
	}
}

void omp_bcodecb_atomic2(PVBCode* codes, size_t n, BCodeCB cb)
{
#pragma omp parallel for
	for (size_t i = 0; i < n; i++) {
		const PVBCode bit = codes[i];
		const uint32_t idx = bcode2cb_idx(bit);
		const uint32_t nbit = bcode2cb_bitn(bit);
		_mm_prefetch(&cb[idx], _MM_HINT_NTA);

#pragma omp atomic
		cb[idx] |= 1<<nbit;
	}
}

void omp_bcodecb(PVBCode* codes, size_t n, BCodeCB cb, BCodeCB* cb_threads, int nth)
{
#pragma omp parallel firstprivate(n) firstprivate(cb_threads) num_threads(nth)
	{
		BCodeCB cb_thread = cb_threads[omp_get_thread_num()];
#pragma omp for
		for (size_t i = 0; i < n; i++) {
			const PVBCode bit = codes[i];
			B_SET(cb_thread[bcode2cb_idx(bit)], bcode2cb_bitn(bit));
		}
	}

	for (int i = 0; i < nth; i++) {
		BCodeCB cb_thread = cb_threads[i];	
		for (uint32_t j = 0; j < NB_INT_BCODECB; j++) {
			cb[j] |= cb_thread[j];
		}
	}
}

void omp_bcodecb_nomerge(PVBCode* codes, size_t n, BCodeCB* cb_threads, int nth)
{
#pragma omp parallel firstprivate(n) firstprivate(cb_threads) num_threads(nth)
	{
		BCodeCB cb_thread = cb_threads[omp_get_thread_num()];
#pragma omp for
		for (size_t i = 0; i < n; i++) {
			const PVBCode bit = codes[i];
			B_SET(cb_thread[bcode2cb_idx(bit)], bcode2cb_bitn(bit));
		}
	}
}

void bcodecb_stream(PVBCode* codes, size_t n, BCodeCB cb)
{
	for (size_t i = 0; i < n; i++) {
		const PVBCode bit = codes[i];
		const uint32_t idx = bcode2cb_idx(bit)>>30;
		const unsigned int v = cb[idx];
		const uint32_t bitn = bcode2cb_bitn(bit);
		cb[idx] |= 1<<(bitn);
	}
}

void bcodecb_sse(PVBCode* codes, size_t n, BCodeCB cb)
{
	__m128i sse_codes;
	__m128i sse_bits;
	const __m128i sse_and31 = _mm_set_epi32(31, 31, 31, 31);
	const size_t end4 = (n/4)*4;
	for (size_t i = 0; i < end4; i += 4) {
		sse_codes = _mm_load_si128((__m128i*) &codes[i]);
		sse_bits = _mm_and_si128(sse_codes, sse_and31);
		sse_codes = _mm_srli_epi32(sse_codes, 5);

		// This un unrolled by GCC!
		for (int i = 0; i < 4; i++) {
			cb[_mm_extract_epi32(sse_codes, i)] |= 1<<(_mm_extract_epi32(sse_bits, i));
		}
	}
	for (size_t i = end4; i < n; i++) {
		PVBCode bit = codes[i];
		//assert_bcode_valid(bit);
		B_SET(cb[(bit.int_v)>>5], ((bit.int_v)&31));
	}
}

void bcodecb_sse_branch(PVBCode* codes, size_t n, BCodeCB cb)
{
	__m128i sse_codes;
	__m128i sse_bits;
	__m128i sse_and31 = _mm_set_epi32(31, 31, 31, 31);
	size_t end4 = (n/4)*4;
	for (size_t i = 0; i < end4; i += 4) {
		sse_codes = _mm_load_si128((__m128i*) &codes[i]);
		sse_bits = _mm_and_si128(sse_codes, sse_and31);
		sse_codes = _mm_srli_epi32(sse_codes, 5);

		const int64_t didxes = _mm_extract_epi64(sse_codes, 0);
		if (didxes == _mm_extract_epi64(sse_codes, 1)) {
			// If all the indices are the same, optimise memory transfers!
			const uint32_t a = 1<<(_mm_extract_epi32(sse_bits, 0)) | 1<<(_mm_extract_epi32(sse_bits, 1));
			const uint32_t b = 1<<(_mm_extract_epi32(sse_bits, 2)) | 1<<(_mm_extract_epi32(sse_bits, 3));
			cb[(uint32_t)didxes] |= a | b;
		}
		else {
			// This un unrolled by GCC!
			for (int i = 0; i < 4; i++) {
				cb[_mm_extract_epi32(sse_codes, i)] |= 1<<(_mm_extract_epi32(sse_bits, i));
			}
		}
	}
	for (size_t i = end4; i < n; i++) {
		PVBCode bit = codes[i];
		//assert_bcode_valid(bit);
		B_SET(cb[(bit.int_v)>>5], ((bit.int_v)&31));
	}
}

void omp_bcodecb_sse_branch(PVBCode* codes, size_t n, BCodeCB cb)
{
	__m128i sse_codes;
	__m128i sse_bits;
	__m128i sse_and31 = _mm_set_epi32(31, 31, 31, 31);
	size_t end4 = (n/4)*4;
#pragma omp parallel for firstprivate(end4) firstprivate(sse_and31) private(sse_codes) private(sse_bits)
	for (size_t i = 0; i < end4; i += 4) {
		sse_codes = _mm_load_si128((__m128i*) &codes[i]);
		sse_bits = _mm_and_si128(sse_codes, sse_and31);
		sse_codes = _mm_srli_epi32(sse_codes, 5);

		const int64_t didxes = _mm_extract_epi64(sse_codes, 0);
		if (didxes == _mm_extract_epi64(sse_codes, 1)) {
			// If all the indices are the same, optimise memory transfers!
			const uint32_t a = 1<<(_mm_extract_epi32(sse_bits, 0)) | 1<<(_mm_extract_epi32(sse_bits, 1));
			const uint32_t b = 1<<(_mm_extract_epi32(sse_bits, 2)) | 1<<(_mm_extract_epi32(sse_bits, 3));
#pragma omp atomic
			cb[(uint32_t)didxes] |= a | b;
		}
		else {
			// This un unrolled by GCC!
			for (int i = 0; i < 4; i++) {
#pragma omp atomic
				cb[_mm_extract_epi32(sse_codes, i)] |= 1<<(_mm_extract_epi32(sse_bits, i));
			}
		}
	}
	for (size_t i = end4; i < n; i++) {
		PVBCode bit = codes[i];
		B_SET(cb[(bit.int_v)>>5], ((bit.int_v)&31));
	}
}

void bcodecb_sse_branch2(PVBCode* codes, size_t n, BCodeCB cb)
{
	__m128i sse_codes;
	__m128i sse_bits;
	const __m128i sse_and31 = _mm_set_epi32(31, 31, 31, 31);
	const size_t end4 = (n/4)*4;
	for (size_t i = 0; i < end4; i += 4) {
		sse_codes = _mm_load_si128((__m128i*) &codes[i]);
		sse_bits = _mm_and_si128(sse_codes, sse_and31);
		sse_codes = _mm_srli_epi32(sse_codes, 5);

		const int64_t didxes0 = _mm_extract_epi64(sse_codes, 0);
		const int64_t didxes1 = _mm_extract_epi64(sse_codes, 1);
		if (didxes0 == didxes1) {
			// If all the indices are the same, optimise memory transfers!
			const uint32_t a = 1<<(_mm_extract_epi32(sse_bits, 0)) | 1<<(_mm_extract_epi32(sse_bits, 1));
			const uint32_t b = 1<<(_mm_extract_epi32(sse_bits, 2)) | 1<<(_mm_extract_epi32(sse_bits, 3));
			cb[(uint32_t)didxes0] |= a | b;
		}
		else {
			const uint32_t idx0 = (uint32_t) didxes0;
			const uint32_t idx2 = (uint32_t) didxes1;

			if (idx0 == idx2) {
				const uint32_t a = 1<<(_mm_extract_epi32(sse_bits, 0)) | 1<<(_mm_extract_epi32(sse_bits, 2));
				cb[idx0] |= a;
				cb[_mm_extract_epi32(sse_codes, 1)] |= 1<<(_mm_extract_epi32(sse_bits, 1));
				cb[_mm_extract_epi32(sse_codes, 3)] |= 1<<(_mm_extract_epi32(sse_bits, 3));
				continue;
			}

			const uint32_t idx1 = (uint32_t)(didxes0>>32);
			const uint32_t idx3 = (uint32_t)(didxes1>>32);
			if (idx1 == idx3) {
				const uint32_t a = 1<<(_mm_extract_epi32(sse_bits, 1)) | 1<<(_mm_extract_epi32(sse_bits, 3));
				cb[idx1] |= a;
				cb[_mm_extract_epi32(sse_codes, 0)] |= 1<<(_mm_extract_epi32(sse_bits, 0));
				cb[_mm_extract_epi32(sse_codes, 2)] |= 1<<(_mm_extract_epi32(sse_bits, 2));
				continue;
			}


			// This un unrolled by GCC!
			for (int i = 0; i < 4; i++) {
				cb[_mm_extract_epi32(sse_codes, i)] |= 1<<(_mm_extract_epi32(sse_bits, i));
			}
		}
	}
	for (size_t i = end4; i < n; i++) {
		PVBCode bit = codes[i];
		//assert_bcode_valid(bit);
		B_SET(cb[(bit.int_v)>>5], ((bit.int_v)&31));
	}
}

void bcodecb_sse_sort_branch(PVBCode* codes, size_t n, BCodeCB cb)
{
	tbb::parallel_sort((uint32_t DECLARE_ALIGN(16) *) codes, ((uint32_t DECLARE_ALIGN(16) *) codes) + n);
	bcodecb_sse_branch(codes, n, cb);
}

void bcodecb_sort_unique(PVBCode_ap codes, size_t n, BCodeCB /*cb*/)
{
	std::sort((uint32_t DECLARE_ALIGN(16) *) codes, ((uint32_t DECLARE_ALIGN(16) *) codes) + n);
	std::unique((uint32_t DECLARE_ALIGN(16) *) codes, ((uint32_t DECLARE_ALIGN(16) *) codes) + n);
}

void bcodecb_parallel_sort_unique(PVBCode_ap codes, size_t n, BCodeCB /*cb*/)
{
	tbb::parallel_sort((uint32_t DECLARE_ALIGN(16) *) codes, ((uint32_t DECLARE_ALIGN(16) *) codes) + n);
	std::unique((uint32_t DECLARE_ALIGN(16) *) codes, ((uint32_t DECLARE_ALIGN(16) *) codes) + n);
}
