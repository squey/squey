/**
 * \file reduction_n64.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

// Goal: find out the performance of a serial reduction vs.
// the size of the destination space !
//
// Same as reduction_n, but w/ 64 bit arrays !

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

#include <common/common.h>
#include <common/bench.h>

#include <pvkernel/core/picviz_intrin.h>
#include <tbb/parallel_sort.h>

//#include "sort/hybrid_sort.h"

#include <omp.h>

#include <time.h>
#include <math.h>
#include <sched.h>

#include <algorithm>

inline bool comp_idx(uint32_t v1, uint32_t v2)
{
	return (v1>>6) < (v2>>6);
}

inline bool comp_4b(uint32_t v1, uint32_t v2)
{
	return (v1>>23) < (v2>>23);
}

struct CCompIdx
{
	CCompIdx(uint32_t* data): _data(data) { }
	inline bool operator()(uint32_t i1, uint32_t i2) const { return _data[i1] < _data[i2]; }

private:
	uint32_t* _data;
};

#define SIZE_INTER_OUT (20*1024)
//#define NB_INT_TILE_BRANCH (128*1024*8)
#define NB_INT_TILE_BRANCH (1024)
typedef uint32_t* DECLARE_ALIGN(16) uint_ap;
typedef uint64_t* DECLARE_ALIGN(16) uint64_ap;

uint32_t _and_mask_d = 0;

void red_ref(size_t n, size_t d, uint_ap in, uint64_ap out)
{
	uint32_t v;
	const uint64_t mask = _and_mask_d;
	for (size_t i = 0; i < n; i++) {
		v = in[i];
		out[(v>>6)] |= 1LL<<(v&63);
	}
}

void red_branch(size_t n, size_t d, uint_ap in, uint64_ap out)
{
	uint32_t v, out_v, idx;
	uint64_t bit_mask;
	for (size_t i = 0; i < n; i++) {
		v = in[i];
		idx = v>>6;
		out_v = out[idx];
		bit_mask = (1LL<<(v&63));
		if ((out_v & bit_mask) == 0) {
			out[idx] = out_v | bit_mask;
		}
	}
}

void red_rsort(size_t n, size_t d, uint_ap in, uint64_ap out)
{
	uint32_t v, out_v, idx;
	uint64_t bit_mask;
	const size_t tend = (n/NB_INT_TILE_BRANCH)*NB_INT_TILE_BRANCH;
	uint32_t DECLARE_ALIGN(16) tile_data[NB_INT_TILE_BRANCH];
	uint32_t DECLARE_ALIGN(16) idxes[NB_INT_TILE_BRANCH];
	for (uint32_t i = 0; i < NB_INT_TILE_BRANCH; i++) {
		idxes[i] = i;
	}
	CCompIdx comp(tile_data);
	BENCH_START(serial_rsort);
	for (size_t t = 0; t < tend; t += NB_INT_TILE_BRANCH) {
		for (size_t i = 0; i < NB_INT_TILE_BRANCH; i++) {
			tile_data[i] = in[t+i]>>6;
		}
		std::sort(idxes, idxes+NB_INT_TILE_BRANCH, comp);
		for (size_t i = 0; i < NB_INT_TILE_BRANCH; i++) {
			uint32_t idx = idxes[i];
			v = in[t+idx];
			idx = tile_data[idx];
			out_v = out[idx];
			bit_mask = (1LL<<(v&63));
			//if ((out_v & bit_mask) == 0) {
				out[idx] = out_v | bit_mask;
			//}
		}
	}
	BENCH_END(serial_rsort, "rsort", n, sizeof(uint32_t), d, sizeof(uint64_t));
}

void red_rsort2(size_t n, size_t d, uint_ap in, uint64_ap out)
{
	uint32_t v, out_v, idx;
	uint64_t bit_mask;
	const size_t tend = (n/NB_INT_TILE_BRANCH)*NB_INT_TILE_BRANCH;
	uint32_t DECLARE_ALIGN(16) tile_data[NB_INT_TILE_BRANCH];
	uint32_t DECLARE_ALIGN(16) idxes[NB_INT_TILE_BRANCH];
	for (uint32_t i = 0; i < NB_INT_TILE_BRANCH; i++) {
		idxes[i] = i;
	}
	BENCH_START(serial_rsort);
	for (size_t t = 0; t < tend; t += NB_INT_TILE_BRANCH) {
		std::sort(idxes, idxes+NB_INT_TILE_BRANCH, CCompIdx(in+t));
		for (size_t i = 0; i < NB_INT_TILE_BRANCH; i++) {
			uint32_t idx = idxes[i];
			v = in[t+idx];
			idx = v>>6;
			out_v = out[idx];
			bit_mask = (1LL<<(v&63));
			//if ((out_v & bit_mask) == 0) {
				out[idx] = out_v | bit_mask;
			//}
		}
	}
	BENCH_END(serial_rsort, "rsort", n, sizeof(uint32_t), d, sizeof(uint64_t));
}

void red_omp_atomic(size_t n, size_t d, uint_ap in, uint64_ap out, int nthreads)
{
#pragma omp parallel for firstprivate(n) num_threads(nthreads)
	for (size_t i = 0; i < n; i++) {
		uint64_t v;
		v = in[i];
		const uint64_t idx = (v>>6);
#pragma omp atomic
		out[idx] |= 1<<(v&63);
	}
}

void red_ref0(size_t n, uint_ap in, uint64_ap out)
{
	uint32_t v;
	uint64_t out0 = 0;
	// This will be vectorized by GCC in -O3...!
	for (size_t i = 0; i < n; i++) {
		v = in[i];
		out0 |= 1<<(v&63);
	}
	out[0] = out0;
}

void red_ref0_sse(size_t n, uint_ap in, uint64_ap out)
{
	const size_t end = (n/8)*8;
	__m128i sse_red = _mm_setzero_si128();
	const __m128i sse_63i = _mm_set1_epi32(63);
	uint64_t out0;
	for (size_t i = 0; i < end; i += 8) {
		__m128i sse_in = _mm_load_si128((__m128i*) &in[i]);
		__m128i sse_res = _mm_and_si128(sse_in, sse_63i);
		sse_red = _mm_or_si128(sse_red, sse_res);
	}
	out0 = _mm_extract_epi64(sse_red, 0) | _mm_extract_epi64(sse_red, 1);
	for (size_t i = end; i < n; i++) {
		out0 |= (in[i])&63;
	}
	out[0] = out0;
}

void init_rand_int(size_t n, uint_ap buf)
{
	for (size_t i = 0; i < n; i++) {
		buf[i] = (((rand()<<1)+1)&(_and_mask_d<<6)) | ((rand())&63);
	}
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		fprintf(stderr, "Usage: %s %d d n\n", argv[0], RAND_MAX);
		return 1;
	}

	size_t d = atoll(argv[1]);
	size_t n = atoll(argv[2]);

	_and_mask_d = ((1<<((uint64_t) (log2f((double)d)))) - 1);

	uint_ap in; uint64_ap out;
	posix_memalign((void**) &in, 16, n*sizeof(uint32_t));
	posix_memalign((void**) &out, 16, d*sizeof(uint64_t));

	srand(time(NULL));
	init_rand_int(n, in);
	memset(out, 0, d*sizeof(uint64_t));

	// Bootstrap openmp
	int a = 0;
#pragma omp parallel for
	for (int i= 0; i < 100000; i++) {
		a += i;
	}

#if 0
	{
		BENCH_START(ref0);
		red_ref0(n, in, out);
		BENCH_END(ref0, "ref0", n, sizeof(uint32_t), 1, sizeof(uint64_t));
		uint64_t v_ref = out[0];

		out[0] = 0;
		BENCH_START(ref0_sse);
		red_ref0_sse(n, in, out);
		BENCH_END(ref0_sse, "ref0-sse", n, sizeof(uint32_t), 1, sizeof(uint64_t));
		CHECK(out[0] == v_ref);
	}

	memset(out, 0, d*sizeof(uint64_t));

	uint_ap in_tmp;
	posix_memalign((void**) &in_tmp, 16, n*sizeof(uint32_t));
	memcpy(in_tmp, in, n*sizeof(uint32_t));
	BENCH_START(sort);
	std::sort(in_tmp, &in_tmp[n]);
	BENCH_END_TRANSFORM(sort, "sort", n, sizeof(uint32_t));

	memcpy(in_tmp, in, n*sizeof(uint32_t));
	BENCH_START(tbb_sort);
	tbb::parallel_sort(in_tmp, &in_tmp[n]);
	BENCH_END_TRANSFORM(tbb_sort, "tbb-sort", n, sizeof(uint32_t));

	memcpy(in_tmp, in, n*sizeof(uint32_t));
	BENCH_START(sort4b);
	tbb::parallel_sort(in_tmp, &in_tmp[n], comp_4b);
	//HybridSort(in_tmp, n);
	BENCH_END_TRANSFORM(sort4b, "sort-4b", n, sizeof(uint32_t));
	free(in_tmp);

	BENCH_START(serial_ref);
	red_ref(n, d, in, out);
	BENCH_END(serial_ref, "ref", n, sizeof(uint32_t), d, sizeof(uint64_t));

	BENCH_START(serial_branch);
	red_branch(n, d, in, out);
	BENCH_END(serial_branch, "branch", n, sizeof(uint32_t), d, sizeof(uint64_t));
	
	red_rsort(n, d, in, out);
	red_rsort2(n, d, in, out);

#endif

	for (int i = 1; i <= 24; i++) {
		memset(out, 0, d*sizeof(uint64_t));
		printf("Num threads: %d\n", i);
		BENCH_START(omp);
		red_omp_atomic(n, d, in, out, i);
		BENCH_END(omp, "omp-atomic", n, sizeof(uint32_t), d, sizeof(uint64_t));
	}

	//red_omp_d2(n, d, in);
#if 0
	red_d2(n, d, in);

	red_omp_d2(n, d, in);
	red_omp_d2(n, d, in);
#endif


	return 0;
}
