#include <common/common.h>
#include <common/bench.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string.h>
#include <smmintrin.h>

void kernel(int* DECLARE_ALIGN(16) res, float* DECLARE_ALIGN(16) f1, float* DECLARE_ALIGN(16) f2, size_t n)
{
	for (size_t i = 0; i < n; i++) {
		res[i] = (int)((f1[i]+f2[i])*2.1f);
	}
}

void kernel_int(int* DECLARE_ALIGN(16) res, int* DECLARE_ALIGN(16) f1, int* DECLARE_ALIGN(16) f2, size_t n)
{
	for (size_t i = 0; i < n; i++) {
		res[i] = (f1[i]+f2[i])*5;
	}
}

void sse_kernel_int(int* DECLARE_ALIGN(16) res, int* DECLARE_ALIGN(16) f1, int* DECLARE_ALIGN(16) f2, size_t n)
{
	__m128i sse_f1, sse_f2, sse_res;
	__m128i sse_cst = _mm_set1_epi32(5);
	for (size_t i = 0; i < n; i += 4) {
		sse_f1 = _mm_load_si128((__m128i*) &f1[i]);
		//sse_f2 = _mm_load_si128((__m128i*) &f2[i]);

		//sse_res = _mm_mul_epi32(_mm_add_epi32(sse_f1, sse_f2), sse_cst);
		sse_res = _mm_mul_epi32(sse_f1, sse_cst);
		_mm_stream_si128((__m128i*) &res[i], sse_res);
	}
}

void omp_kernel(int* DECLARE_ALIGN(16) res, float* DECLARE_ALIGN(16) f1, float* DECLARE_ALIGN(16) f2, size_t n)
{
#pragma omp parallel for num_threads(12) schedule(static)
	for (size_t i = 0; i < n; i++) {
		res[i] = (int)((f1[i]+f2[i])*2.1f);
	}
}

void sse_kernel(int* DECLARE_ALIGN(16) res, float* DECLARE_ALIGN(16) f1, float* DECLARE_ALIGN(16) f2, size_t n)
{
	__m128 sse_cst = _mm_set1_ps(2.1f);
	for (size_t i = 0; i < n; i += 4) {
		__m128 sse_f1 = _mm_load_ps(&f1[i]);
		__m128 sse_f2 = _mm_load_ps(&f2[i]);

		__m128 sse_add = _mm_add_ps(sse_f1, sse_f2);
		__m128 sse_mul = _mm_mul_ps(sse_add, sse_cst);
		__m128i sse_res = _mm_cvtps_epi32(sse_mul);

		_mm_stream_si128((__m128i*) &res[i], sse_res);
	}
	_mm_sfence();
}

void sse_prefetch_kernel(int* DECLARE_ALIGN(16) res, float* DECLARE_ALIGN(16) f1, float* DECLARE_ALIGN(16) f2, size_t n)
{
	__m128 sse_cst = _mm_set1_ps(2.1f);
	for (size_t i = 0; i < n; i += 4) {
		__m128 sse_f1 = _mm_load_ps(&f1[i]);
		__m128 sse_f2 = _mm_load_ps(&f2[i]);
		_mm_prefetch(&f1[i+4], _MM_HINT_NTA);
		_mm_prefetch(&f2[i+4], _MM_HINT_NTA);

		__m128 sse_add = _mm_add_ps(sse_f1, sse_f2);
		__m128 sse_mul = _mm_mul_ps(sse_add, sse_cst);
		__m128i sse_res = _mm_cvtps_epi32(sse_mul);

		_mm_stream_si128((__m128i*) &res[i], sse_res);
	}
	_mm_sfence();
}

void omp_sse_kernel(int* DECLARE_ALIGN(16) res, float* DECLARE_ALIGN(16) f1, float* DECLARE_ALIGN(16) f2, size_t n)
{
	__m128 sse_cst = _mm_set1_ps(2.1f);
#pragma omp parallel for num_threads(12) schedule(static)
	for (size_t i = 0; i < n; i += 4) {
		__m128 sse_f1 = _mm_load_ps(&f1[i]);
		__m128 sse_f2 = _mm_load_ps(&f2[i]);

		__m128 sse_add = _mm_add_ps(sse_f1, sse_f2);
		__m128 sse_mul = _mm_mul_ps(sse_add, sse_cst);
		__m128i sse_res = _mm_cvtps_epi32(sse_mul);

		_mm_stream_si128((__m128i*) &res[i], sse_res);
	}
	_mm_sfence();
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		fprintf(stderr, "Usage: %s n\n", argv[0]);
		return 1;
	}
	srand(time(NULL));

	size_t n = atoll(argv[1]);

	float DECLARE_ALIGN(16) *f1, DECLARE_ALIGN(16) *f2;
	int* DECLARE_ALIGN(16) res;
	int* DECLARE_ALIGN(16) res_ref;

	posix_memalign((void**) &f1, 16, sizeof(float)*n);
	posix_memalign((void**) &f2, 16, sizeof(float)*n);
	posix_memalign((void**) &res, 16, sizeof(int)*n);
	posix_memalign((void**) &res_ref, 16, sizeof(int)*n);
	
	for (size_t i = 0; i < n; i++) {
		f1[i] = (float) rand();
		f2[i] = (float) rand();
	}

	BENCH_START(k);
	kernel(res_ref, f1, f2, n);
	BENCH_END(k, "ref", n*2, sizeof(float), n, sizeof(int));

	BENCH_START(k2);
	kernel_int(res_ref, (int*) f1, (int*) f2, n);
	BENCH_END(k2, "ref-int", n*2, sizeof(float), n, sizeof(int));

	BENCH_START(ki);
	sse_kernel_int(res_ref, (int*) f1, (int*) f2, n);
	BENCH_END(ki, "sse-int", n, sizeof(float), n, sizeof(int));
	
	BENCH_START(omp);
	omp_kernel(res, f1, f2, n);
	BENCH_END(omp, "omp", n*2, sizeof(float), n, sizeof(int));
	CHECK(memcmp(res, res_ref, sizeof(int)*n) == 0);

	BENCH_START(sse);
	sse_kernel(res, f1, f2, n);
	BENCH_END(sse, "sse", n*2, sizeof(float), n, sizeof(int));
	CHECK(memcmp(res, res_ref, sizeof(int)*n) == 0);

	BENCH_START(ssep);
	sse_prefetch_kernel(res, f1, f2, n);
	BENCH_END(ssep, "sse-prefetch", n*2, sizeof(float), n, sizeof(int));
	CHECK(memcmp(res, res_ref, sizeof(int)*n) == 0);

	BENCH_START(sseomp);
	omp_sse_kernel(res, f1, f2, n);
	BENCH_END(sseomp, "sse-omp", n*2, sizeof(float), n, sizeof(int));
	CHECK(memcmp(res, res_ref, sizeof(int)*n) == 0);

	return 0;
}
