// Goal: find out the performance of a serial reduction vs.
// the size of the destination space !
//
// Same as reduction_n, but w/ 64 bit arrays !

#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

#include <common/common.h>
#include <common/bench.h>

#include <pvkernel/core/picviz_intrin.h>

#include <omp.h>

#include <time.h>
#include <math.h>
#include <sched.h>

#define SIZE_INTER_OUT (20*1024)
typedef uint64_t* DECLARE_ALIGN(16) uint_ap;

uint64_t _and_mask_d = 0;

void red_ref(size_t n, size_t d, uint_ap in, uint_ap out)
{
	uint64_t v;
	const uint64_t mask = _and_mask_d;
	for (size_t i = 0; i < n; i++) {
		v = in[i];
		out[(v>>6)&mask] |= v&63;
	}
}

void red_ref0(size_t n, uint_ap in, uint_ap out)
{
	uint64_t v;
	uint64_t out0 = 0;
	// This will be vectorized by GCC in -O3...!
	for (size_t i = 0; i < n; i++) {
		v = in[i];
		out0 |= v&63;
	}
	out[0] = out0;
}

void red_ref0_sse(size_t n, uint_ap in, uint_ap out)
{
	const size_t end = (n/8)*8;
	__m128i sse_red = _mm_setzero_si128();
	const __m128i sse_63i = _mm_set_epi32(63, 0, 63, 0);
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
		buf[i] = (rand()<<1)+1;
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

	uint_ap in,out;
	posix_memalign((void**) &in, 16, n*sizeof(uint64_t));
	posix_memalign((void**) &out, 16, d*sizeof(uint64_t));

	srand(time(NULL));
	init_rand_int(n, in);
	memset(out, 0, d*sizeof(uint64_t));

	{
		BENCH_START(ref0);
		red_ref0(n, in, out);
		BENCH_END(ref0, "ref0", n, sizeof(uint64_t), 1, sizeof(uint64_t));
		uint64_t v_ref = out[0];

		out[0] = 0;
		BENCH_START(ref0_sse);
		red_ref0_sse(n, in, out);
		BENCH_END(ref0_sse, "ref0-sse", n, sizeof(uint64_t), 1, sizeof(uint64_t));
		CHECK(out[0] == v_ref);
	}

	BENCH_START(serial_ref);
	red_ref(n, d, in, out);
	BENCH_END(serial_ref, "ref", n, sizeof(uint64_t), d, sizeof(uint64_t));

	//red_omp_d2(n, d, in);
#if 0
	red_d2(n, d, in);

	red_omp_d2(n, d, in);
	red_omp_d2(n, d, in);
#endif


	return 0;
}
