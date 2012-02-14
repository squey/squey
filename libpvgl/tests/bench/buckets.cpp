#include <common/bench.h>
#include <common/common.h>

#include <iostream>

#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <pvkernel/core/picviz_intrin.h>

typedef uint32_t DECLARE_ALIGN(16) * uint_ap;

void buckets_ref(uint_ap in, const size_t n, uint_ap buckets, const size_t nbuckets, const size_t nbuckets_ln2, const uint16_t bucket_size)
{
	const uint8_t nbits_shift = 32-nbuckets_ln2;
	uint16_t buckets_pos[nbuckets];
	memset(buckets_pos, 0, sizeof(uint16_t)*nbuckets);
	for (size_t i = 0; i < n; i++) {
		uint32_t v = in[i];
		uint32_t idx_bucket = v>>nbits_shift;
		uint16_t pos_bucket = buckets_pos[idx_bucket];
		buckets[idx_bucket*bucket_size+pos_bucket] = v;
		if (pos_bucket == bucket_size-1) {
			pos_bucket = 0;
		}
		else {
			pos_bucket++;
		}
		buckets_pos[idx_bucket] = pos_bucket;
	}
}

void buckets_ref2(uint_ap in, const size_t n, uint_ap buckets, const size_t nbuckets, const size_t nbuckets_ln2, const uint16_t bucket_size)
{
	const uint8_t nbits_shift = 32-nbuckets_ln2;
	uint16_t buckets_pos[nbuckets];
	memset(buckets_pos, 0, sizeof(uint16_t)*nbuckets);
	/*
	for (size_t i = 0; i < (nbuckets/4)*4; i += 4) {
		_mm_prefetch(&buckets_pos[i], _MM_HINT_NTA);
	}*/
	for (size_t i = 0; i < n; i++) {
		uint32_t v = in[i];
		uint32_t idx_bucket = v>>nbits_shift;
		uint16_t* ppos_bucket = &buckets_pos[idx_bucket];
		uint16_t pos_bucket = *ppos_bucket;
		buckets[idx_bucket*bucket_size+pos_bucket] = v;
		if (pos_bucket == bucket_size-1) {
			pos_bucket = 0;
		}
		else {
			pos_bucket++;
		}
		*ppos_bucket = pos_bucket;
	}
}

void buckets_posheap(uint_ap in, const size_t n, uint_ap buckets, const size_t nbuckets, const size_t nbuckets_ln2, const uint16_t bucket_size)
{
	const uint8_t nbits_shift = 32-nbuckets_ln2;
	uint16_t DECLARE_ALIGN(16) *buckets_pos;
	posix_memalign((void**) &buckets_pos, 16, sizeof(uint16_t)*nbuckets);
	memset(buckets_pos, 0, sizeof(uint16_t)*nbuckets);
	for (size_t i = 0; i < n; i++) {
		uint32_t v = in[i];
		uint32_t idx_bucket = v>>nbits_shift;
		uint16_t* ppos_bucket = &buckets_pos[idx_bucket];
		uint16_t pos_bucket = *ppos_bucket;
		buckets[idx_bucket*bucket_size+pos_bucket] = v;
		if (pos_bucket == bucket_size-1) {
			pos_bucket = 0;
		}
		else {
			pos_bucket++;
		}
		*ppos_bucket = pos_bucket;
	}
}

void buckets_sse2(uint_ap in, const size_t n, uint_ap buckets, const size_t nbuckets, const size_t nbuckets_ln2, const uint16_t bucket_size)
{
	const uint8_t nbits_shift = 32-nbuckets_ln2;
	uint16_t buckets_pos[nbuckets];
	memset(buckets_pos, 0, sizeof(uint16_t)*nbuckets);
	const size_t nend = (n/4)*4;
	__m128i sse_v, sse_idx_bucket;
	__m128i sse_pos_bucket;
	const __m128i sse_1i = _mm_set1_epi32(1);
	const __m128i sse_bucket_size = _mm_set1_epi32(bucket_size);
	const __m128i sse_ff = _mm_set1_epi32(0xffffffff);
	for (size_t i = 0; i < nend; i += 4) {
		sse_v = _mm_load_si128((__m128i*) &in[i]);
		sse_idx_bucket = _mm_srli_epi32(sse_v, nbits_shift);
		for (int i = 0; i < 4; i++) {
			const uint32_t idx_bucket = _mm_extract_epi32(sse_idx_bucket, i);
			const uint32_t pos_bucket = buckets_pos[idx_bucket];
			buckets[idx_bucket*bucket_size+pos_bucket] = _mm_extract_epi32(sse_v, i);
			_mm_insert_epi32(sse_pos_bucket, pos_bucket, i);
		}
		sse_pos_bucket = _mm_add_epi32(sse_pos_bucket, sse_1i);
		__m128i sse_cmp_pos = _mm_cmpeq_epi32(sse_pos_bucket, sse_bucket_size);
		if (!_mm_testz_si128(sse_cmp_pos, sse_ff)) {
			// One or more of the bucket are full. Set its pos to 0.
			sse_pos_bucket = _mm_andnot_si128(sse_cmp_pos, sse_pos_bucket);
			for (int i = 0; i < 4; i++) {
				buckets_pos[_mm_extract_epi32(sse_idx_bucket, i)] = _mm_extract_epi32(sse_pos_bucket, i);
			}
		}
	}
}

void init_rand_int(size_t n, uint_ap buf)
{
	for (size_t i = 0; i < n; i++) {
		buf[i] = ((rand()<<1)+1);
	}
}

int main(int argc, char** argv)
{
	if (argc < 4) {
		std::cerr << "Usage: " << argv[0] << " nint nbuckets_log2 size_bucket" << std::endl;
		return 1;
	}

	size_t n = atoll(argv[1]);
	size_t nbuckets_ln2 = atoll(argv[2]);
	uint16_t bucket_size = atoi(argv[3]);
	size_t nbuckets = 1<<(nbuckets_ln2);

	srand(time(NULL));

	uint_ap in;
	posix_memalign((void**) &in, 16, n*sizeof(uint32_t));
	init_rand_int(n, in);

	uint32_t* buckets[nbuckets];
	printf("Buckets size: %f KB\n", ((double)(nbuckets*bucket_size*sizeof(uint32_t)))/1024.0);
	posix_memalign((void**) &buckets[0], 16, nbuckets*bucket_size*sizeof(uint32_t));
	memset(buckets[0], 0, nbuckets*bucket_size*sizeof(uint32_t));
	for (size_t i = 1; i < nbuckets; i++) {
		buckets[i] = buckets[0] + i*bucket_size;
	}

	BENCH_START(ref);
	buckets_ref(in, n, buckets[0], nbuckets, nbuckets_ln2, bucket_size);
	BENCH_END(ref, "ref", n, sizeof(uint32_t), n, sizeof(uint32_t));

	BENCH_START(ref2);
	buckets_ref2(in, n, buckets[0], nbuckets, nbuckets_ln2, bucket_size);
	BENCH_END(ref2, "ref2", n, sizeof(uint32_t), n, sizeof(uint32_t));

	/*
	BENCH_START(ref_posheap);
	buckets_posheap(in, n, buckets[0], nbuckets, nbuckets_ln2, bucket_size);
	BENCH_END(ref_posheap, "posheap", n, sizeof(uint32_t), n, sizeof(uint32_t));

	BENCH_START(sse2);
	buckets_sse2(in, n, buckets[0], nbuckets, nbuckets_ln2, bucket_size);
	BENCH_END(sse2, "sse2", n, sizeof(uint32_t), n, sizeof(uint32_t));*/
	

	return 0;
}
