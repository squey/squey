// Goal: find out the performance of a serial reduction vs.
// the size of the destination space !
//
// Let's define:
//   * in[N] the source space
//   * out[D] the destination space.
//
// The reduction will :
//  * take as input a buffer of int32_t integers (in[N])
//  * and will create :
//      for (i: 0->N):
//          v = i
//          out[(i>>5)%D] |= i&31
//
// Let's do it...

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

#include <common/common.h>
#include <common/bench.h>

#include <pvkernel/core/picviz_intrin.h>

#include <omp.h>

#include <time.h>
#include <math.h>

#define SIZE_INTER_OUT (16*1024)
typedef uint32_t* DECLARE_ALIGN(16) uint_ap;

uint32_t _and_mask_d = 0;

void red_ref(size_t n, size_t d, uint_ap in, uint_ap out)
{
	uint32_t v;
	const uint32_t mask = _and_mask_d;
	for (size_t i = 0; i < n; i++) {
		v = in[i];
		out[(v>>5)&mask] |= v&31;
	}
}

void red_d2(size_t n, size_t d, uint_ap in)
{
	uint32_t v;
	uint32_t* outs[2];
	posix_memalign((void**) &outs[0], 16, sizeof(uint32_t)*SIZE_INTER_OUT);
	posix_memalign((void**) &outs[1], 16, sizeof(uint32_t)*SIZE_INTER_OUT);
	memset(outs[0], 0, SIZE_INTER_OUT);
	memset(outs[1], 0, SIZE_INTER_OUT);
	const uint32_t mask = _and_mask_d;
	BENCH_START(bench);
	for (size_t i = 0; i < n; i++) {
		v = in[i];
		const uint32_t idx = (v>>5)&mask;
		const uint32_t vor = v&31;
		if (idx < SIZE_INTER_OUT) {
			outs[0][idx] |= vor;
		}
		else {
			printf("%u\n", idx-SIZE_INTER_OUT);
			outs[1][idx-SIZE_INTER_OUT] |= vor;
		}
	}
	BENCH_END(bench, "ref-d2", n, sizeof(uint32_t), 2*SIZE_INTER_OUT, sizeof(uint32_t));
}

void red_omp_d2(size_t n, size_t d, uint_ap in)
{
	uint32_t v;
	const uint32_t mask = _and_mask_d;
	uint32_t* outs[2];
	posix_memalign((void**) &outs[0], 16, sizeof(uint32_t)*SIZE_INTER_OUT);
	posix_memalign((void**) &outs[1], 16, sizeof(uint32_t)*SIZE_INTER_OUT);
	memset(outs[0], 0, SIZE_INTER_OUT);
	memset(outs[1], 0, SIZE_INTER_OUT);
	BENCH_START(bench);
#pragma omp parallel sections
	{
#pragma omp section
		{
			uint_ap out = outs[0];
			for (size_t i = 0; i < n; i++) {
				v = in[i];
				const uint32_t idx = (v>>5)&mask;
				if (idx < SIZE_INTER_OUT) {
					out[idx] |= v&31;
				}
			}
		}
#pragma omp section
		{
			uint_ap out = outs[1];
			for (size_t i = 0; i < n; i++) {
				v = in[i];
				const uint32_t idx = (v>>5)&mask;
				if (idx >= SIZE_INTER_OUT) {
					out[idx-SIZE_INTER_OUT] |= v&31;
				}
			}
		}
	}
	BENCH_END(bench, "omp", n, sizeof(uint32_t), 2*SIZE_INTER_OUT, sizeof(uint32_t));
}

void red_ref0(size_t n, uint_ap in, uint_ap out)
{
	uint32_t v;
	uint32_t out0 = 0;
	// This will be vectorized by GCC in -O3...!
	for (size_t i = 0; i < n; i++) {
		v = in[i];
		out0 |= v&31;
	}
	out[0] = out0;
}

void red_ref0_sse(size_t n, uint_ap in, uint_ap out)
{
	const size_t end = (n/4)*4;
	__m128i sse_red = _mm_setzero_si128();
	const __m128i sse_31i = _mm_set1_epi32(31);
	uint32_t out0;
	for (size_t i = 0; i < end; i += 4) {
		__m128i sse_in = _mm_load_si128((__m128i*) &in[i]);
		__m128i sse_res = _mm_and_si128(sse_in, sse_31i);
		sse_red = _mm_or_si128(sse_red, sse_res);
	}
	out0 = _mm_extract_epi32(sse_red, 0) | _mm_extract_epi32(sse_red, 1) | _mm_extract_epi32(sse_red, 2) | _mm_extract_epi32(sse_red, 3);
	for (size_t i = end; i < n; i++) {
		out0 |= (in[i])&31;
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

	_and_mask_d = ((1<<((uint32_t) (log2f((double)d)))) - 1);

	uint_ap in,out;
	posix_memalign((void**) &in, 16, n*sizeof(uint32_t));
	posix_memalign((void**) &out, 16, d*sizeof(uint32_t));

	srand(time(NULL));
	init_rand_int(n, in);
	memset(out, 0, d*sizeof(uint32_t));

#if 0
	{
		BENCH_START(ref0);
		red_ref0(n, in, out);
		BENCH_END(ref0, "ref0", n, sizeof(uint32_t), 1, sizeof(uint32_t));
		uint32_t v_ref = out[0];

		out[0] = 0;
		BENCH_START(ref0_sse);
		red_ref0_sse(n, in, out);
		BENCH_END(ref0_sse, "ref0-sse", n, sizeof(uint32_t), 1, sizeof(uint32_t));
		CHECK(out[0] == v_ref);
	}
#endif

	BENCH_START(serial_ref);
	red_ref(n, d, in, out);
	BENCH_END(serial_ref, "ref", n, sizeof(uint32_t), d, sizeof(uint32_t));

	red_d2(n, d, in);

	red_omp_d2(n, d, in);
	red_omp_d2(n, d, in);


	return 0;
}
