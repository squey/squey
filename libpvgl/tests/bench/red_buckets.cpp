#include <common/bench.h>
#include <common/common.h>

#include <iostream>

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include <pvkernel/core/picviz_intrin.h>

#define BUCKET_END (0x80000000)

typedef uint32_t DECLARE_ALIGN(16) * uint_ap;
typedef uint64_t DECLARE_ALIGN(16) * uint64_ap;

uint32_t _and_mask_d = 0;

void red_ref(size_t n, size_t d, uint_ap in, uint64_ap out)
{
	uint32_t v;
	for (size_t i = 0; i < n; i++) {
		v = in[i];
		out[(v>>6)] |= 1LL<<(v&63);
	}
}

template <typename bucket_v_t>
void red_ref_bucket(size_t n, bucket_v_t* in, uint64_ap out)
{
	bucket_v_t v;
	for (size_t i = 0; i < n; i++) {
		v = in[i];
		out[(v>>6)] |= 1LL<<(v&63);
	}
}

template <typename bucket_v_t>
void red_ref_bucket_stream(size_t n, bucket_v_t* in, uint64_ap out, uint64_ap out_cache, const size_t sout_cache)
{
	bucket_v_t v;
	memset(out_cache, 0, sout_cache*sizeof(uint64_t));
	for (size_t i = 0; i < n; i++) {
		v = in[i];
		out_cache[(v>>6)] |= 1LL<<(v&63);
	}
	/*
	 * AG: I think I'm too tired, because this give incorrect results... :/
	__m128i sse_out;
	__m128i sse_out_cache;
	for (size_t i = 0; i < (sout_cache/4)*4; i += 4) {
		sse_out_cache = _mm_load_si128((__m128i*) &out_cache[i]);
		sse_out = _mm_or_si128(_mm_load_si128((__m128i*) &out[i]), sse_out_cache);
		_mm_store_si128((__m128i*) &out[i], sse_out);
	}*/
	//memcpy(out, out_cache, sout_cache*sizeof(uint64_t));
	for (size_t i = 0; i < sout_cache; i++) {
		out[i] |= out_cache[i];
	}
}

template <typename bucket_pos_t, typename bucket_v_t>
void red_buckets(const size_t n, const size_t nbits_d, uint_ap in, uint64_ap out, bucket_v_t* buckets, const size_t nbuckets, const size_t nbuckets_ln2, const uint8_t bucket_size_ln2)
{
	const uint8_t nbits_shift = (nbits_d+6)-nbuckets_ln2;
	if (nbits_shift > sizeof(bucket_v_t)*8) {
		printf("error: nbits_shift > %lu.\n", sizeof(bucket_v_t)*8);
		return;
	}
	const bucket_v_t mask_v = (1<<nbits_shift) - 1;
	bucket_pos_t buckets_pos[nbuckets];
	memset(buckets_pos, 0, sizeof(bucket_pos_t)*nbuckets);
	uint32_t v;
	const bucket_pos_t bucket_size = (1<<bucket_size_ln2);
	BENCH_START(red_buckets)
	for (size_t i = 0; i < n; i++) {
		v = in[i];
		const uint32_t idx_bucket = v>>nbits_shift;
		bucket_pos_t* ppos_bucket = &buckets_pos[idx_bucket];
		bucket_pos_t pos_bucket = *ppos_bucket;
		bucket_v_t* b = &buckets[idx_bucket<<bucket_size_ln2];
		b[pos_bucket] = (bucket_v_t) v;
		//b[pos_bucket] = v&mask_v;
		if (pos_bucket == bucket_size-1) {
			red_ref_bucket(bucket_size, b, out);
			pos_bucket = 0;
		}
		else {
			pos_bucket++;
		}
		*ppos_bucket = pos_bucket;
	}
	for (size_t i = 0; i < nbuckets; i++) {
		bucket_pos_t pos_bucket = buckets_pos[i];
		if (pos_bucket > 0) {
			red_ref_bucket(pos_bucket, &buckets[i<<bucket_size_ln2], out);
		}
	}
	BENCH_END(red_buckets, "red-buckets-nocommit", n, sizeof(uint32_t), 0, sizeof(uint64_t));
}

template <typename bucket_pos_t, typename bucket_v_t>
void red_buckets_stream(const size_t n, const size_t nbits_d, uint_ap in, uint64_ap out, bucket_v_t* buckets, const size_t nbuckets, const size_t nbuckets_ln2, const uint8_t bucket_size_ln2)
{
	const uint8_t nbits_shift = (nbits_d+6)-nbuckets_ln2;
	if (nbits_shift > sizeof(bucket_v_t)*8) {
		printf("error: nbits_shift > %lu.\n", sizeof(bucket_v_t)*8);
		return;
	}
	const bucket_v_t mask_v = (1<<nbits_shift) - 1;
	bucket_pos_t buckets_pos[nbuckets];
	memset(buckets_pos, 0, sizeof(bucket_pos_t)*nbuckets);
	uint32_t v;
	const bucket_pos_t bucket_size = (1<<bucket_size_ln2);
	const size_t sout_cache = (1<<(nbits_d-nbuckets_ln2));
	if (sout_cache % 4 != 0) {
		printf("error: sout_cache %% 4 != 0 !\n");
		return;
	}
	uint64_ap out_cache;
	posix_memalign((void**) &out_cache, 16, sout_cache*sizeof(uint64_t));
	BENCH_START(red_buckets)
	for (size_t i = 0; i < n; i++) {
		v = in[i];
		const uint32_t idx_bucket = v>>nbits_shift;
		bucket_pos_t* ppos_bucket = &buckets_pos[idx_bucket];
		bucket_pos_t pos_bucket = *ppos_bucket;
		bucket_v_t* b = &buckets[idx_bucket<<bucket_size_ln2];
		b[pos_bucket] = ((bucket_v_t) v)&mask_v;
		if (pos_bucket == bucket_size-1) {
			red_ref_bucket_stream(bucket_size, b, &out[idx_bucket<<(nbits_d-nbuckets_ln2)], out_cache, sout_cache);
			pos_bucket = 0;
		}
		else {
			pos_bucket++;
		}
		*ppos_bucket = pos_bucket;
	}
	for (size_t i = 0; i < nbuckets; i++) {
		bucket_pos_t pos_bucket = buckets_pos[i];
		if (pos_bucket > 0) {
			red_ref_bucket_stream(pos_bucket, &buckets[i<<bucket_size_ln2], &out[i<<(nbits_d-nbuckets_ln2)], out_cache, sout_cache);
		}
	}
	BENCH_END(red_buckets, "red-buckets", n, sizeof(uint32_t), 0, sizeof(uint64_t));
}

void init_rand_int(size_t n, uint_ap buf)
{
	for (size_t i = 0; i < n; i++) {
		uint32_t v = (rand())&_and_mask_d;
		buf[i] = v;
	}
}

int main(int argc, char** argv)
{
	if (argc < 5) {
		std::cerr << "Usage: " << argv[0] << " dint nint nbuckets_log2 size_bucket_ln2" << std::endl;
		return 1;
	}

	size_t d = atoll(argv[1]);
	size_t n = atoll(argv[2]);
	uint16_t nbuckets_ln2 = atoi(argv[3]);
	uint8_t bucket_size_ln2 = atoi(argv[4]);
	uint32_t bucket_size = 1<<bucket_size_ln2;
	uint16_t nbuckets = 1<<(nbuckets_ln2);

	uint32_t nbits_d = (uint32_t) (log2f((double)d));
	printf("nbits_d: %d\n", nbits_d);
	_and_mask_d = ((1<<(nbits_d+6)) - 1);

	srand(time(NULL));

	uint_ap in; uint64_ap out; uint64_ap out_ref;
	posix_memalign((void**) &in, 16, n*sizeof(uint32_t));
	posix_memalign((void**) &out, 16, d*sizeof(uint64_t));
	posix_memalign((void**) &out_ref, 16, d*sizeof(uint64_t));
	memset(out, 0, d*sizeof(uint64_t));
	memset(out_ref, 0, d*sizeof(uint64_t));
	init_rand_int(n, in);

	uint32_t* buckets;
	size_t sbuckets = (nbuckets)*(bucket_size)*sizeof(uint32_t);
	printf("Buckets size: %f KB\n", ((double)sbuckets)/1024.0);
	posix_memalign((void**) &buckets, 16, sbuckets);
	memset(buckets, 0, sbuckets);
	
	BENCH_START(red_ref);
	red_ref(n, d, in, out_ref);
	BENCH_END(red_ref, "red-ref", n, sizeof(uint32_t), d, sizeof(uint64_t));

	//if (bucket_size_ln2 < 16)
	//	red_buckets<uint16_t, uint32_t>(n, nbits_d, in, out, buckets, nbuckets, nbuckets_ln2, bucket_size_ln2);
	memset(out, 0, d*sizeof(uint64_t));
	red_buckets<uint32_t, uint32_t>(n, nbits_d, in, out, buckets, nbuckets, nbuckets_ln2, bucket_size_ln2);
	CHECK(memcmp(out_ref, out, d*sizeof(uint64_t)) == 0);
	//if (bucket_size_ln2 < 16)
	//	red_buckets_stream<uint16_t, uint32_t>(n, nbits_d, in, out, buckets, nbuckets, nbuckets_ln2, bucket_size_ln2);
	memset(out, 0, d*sizeof(uint64_t));
	red_buckets_stream<uint32_t, uint32_t>(n, nbits_d, in, out, buckets, nbuckets, nbuckets_ln2, bucket_size_ln2);
	CHECK(memcmp(out_ref, out, d*sizeof(uint64_t)) == 0);

	if (nbuckets_ln2 >= 9) {
		printf("Using 16-bit for the bucket buffers...\n");
		memset(out, 0, d*sizeof(uint64_t));
		red_buckets<uint32_t, uint16_t>(n, nbits_d, in, out, (uint16_t*) buckets, nbuckets, nbuckets_ln2, bucket_size_ln2);
		CHECK(memcmp(out_ref, out, d*sizeof(uint64_t)) == 0);
		memset(out, 0, d*sizeof(uint64_t));
		red_buckets_stream<uint32_t, uint16_t>(n, nbits_d, in, out, (uint16_t*) buckets, nbuckets, nbuckets_ln2, bucket_size_ln2);
		CHECK(memcmp(out_ref, out, d*sizeof(uint64_t)) == 0);
	}

	return 0;
}
