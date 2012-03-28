#include <common/bench.h>
#include <common/common.h>

#include <iostream>

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include <set>
#include <tbb/parallel_sort.h>

#include <omp.h>

#include <pvkernel/core/picviz_intrin.h>

#define BUCKET_END (0x80000000)
#define MASK_BUCKET_POS (0x7fffffff)

typedef uint32_t DECLARE_ALIGN(16) * uint_ap;
typedef uint64_t DECLARE_ALIGN(16) * uint64_ap;
typedef uint8_t DECLARE_ALIGN(16) * uint8_ap;

uint32_t _and_mask_d = 0;

void red_ref(size_t n, size_t d, uint_ap in, uint64_ap out)
{
	uint32_t v;
	for (size_t i = 0; i < n; i++) {
		v = in[i];
		out[(v>>6)] |= 1LL<<(v&63);
	}
}

void red_ref_8(size_t n, size_t d, uint_ap in, uint8_ap out)
{
	uint32_t v;
	for (size_t i = 0; i < n; i++) {
		v = in[i];
		out[v] = 1;
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
void red_bucket_commit_clear(size_t n, bucket_v_t* in, uint64_ap out, uint64_ap out_cache, const size_t sout_cache)
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
	for (size_t i = 0; i < sout_cache; i++) {
		out[i] |= out_cache[i];
	}
}

template <typename bucket_v_t>
void red_bucket_nocommit_noclear(size_t n, bucket_v_t* in, uint64_ap out, uint64_ap out_cache, const size_t sout_cache)
{
	bucket_v_t v;
	for (size_t i = 0; i < n; i++) {
		v = in[i];
		out_cache[(v>>6)] |= 1LL<<(v&63);
	}
}


template <typename bucket_pos_t, typename bucket_v_t, typename idx_bucket_t>
void red_buckets(const size_t n, const size_t nbits_d, uint_ap in, uint64_ap out, bucket_v_t* buckets, const size_t nbuckets_ln2, const uint8_t bucket_size_ln2)
{
	const uint8_t nbits_shift = (nbits_d+6)-nbuckets_ln2;
	if (nbits_shift > sizeof(bucket_v_t)*8) {
		printf("error: nbits_shift > %lu.\n", sizeof(bucket_v_t)*8);
		return;
	}
	const idx_bucket_t nbuckets = 1<<nbuckets_ln2;
	const bucket_v_t mask_v = (1<<nbits_shift) - 1;
	bucket_pos_t buckets_pos[nbuckets];
	memset(buckets_pos, 0, sizeof(bucket_pos_t)*nbuckets);
	uint32_t v;
	const bucket_pos_t bucket_size = (1<<bucket_size_ln2);
	BENCH_START(red_buckets)
	for (size_t i = 0; i < n; i++) {
		v = in[i];
		const idx_bucket_t idx_bucket = v>>nbits_shift;
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
	for (idx_bucket_t i = 0; i < nbuckets; i++) {
		bucket_pos_t pos_bucket = buckets_pos[i];
		if (pos_bucket > 0) {
			red_ref_bucket(pos_bucket, &buckets[i<<bucket_size_ln2], out);
		}
	}
	BENCH_END(red_buckets, "red-buckets", n, sizeof(uint32_t), 0, sizeof(uint64_t));
}

template <typename bucket_pos_t, typename bucket_v_t, typename idx_bucket_t>
void red_buckets_ordered(const size_t n, const size_t nbits_d, uint_ap in, uint64_ap out, bucket_v_t* buckets, const size_t nbuckets_ln2, const uint8_t bucket_size_ln2)
{
	// Last bit of bucket_pos_t is reserved for knowing if it is half full set
	const uint8_t nbits_shift = (nbits_d+6)-nbuckets_ln2;
	if (nbits_shift > sizeof(bucket_v_t)*8) {
		printf("error: nbits_shift > %lu.\n", sizeof(bucket_v_t)*8);
		return;
	}
	const idx_bucket_t nbuckets = 1<<nbuckets_ln2;
	const bucket_v_t mask_v = (1<<nbits_shift) - 1;
	bucket_pos_t buckets_pos[nbuckets];
	memset(buckets_pos, 0, sizeof(bucket_pos_t)*nbuckets);
	uint32_t v;
	const bucket_pos_t bucket_size = (1<<bucket_size_ln2);
	const bucket_pos_t mask_half_full = 1<<(bucket_size_ln2-1);
	std::set<idx_bucket_t> hfull_buckets;
	BENCH_START(red_buckets)
	for (size_t i = 0; i < n; i++) {
		v = in[i];
		const idx_bucket_t idx_bucket = v>>nbits_shift;
		bucket_pos_t* ppos_bucket = &buckets_pos[idx_bucket];
		bucket_pos_t pos_bucket = *ppos_bucket;
		const bucket_pos_t real_pos_bucket = pos_bucket & MASK_BUCKET_POS;
		bucket_v_t* b = &buckets[idx_bucket<<bucket_size_ln2];
		b[real_pos_bucket] = (bucket_v_t) v;
		if (real_pos_bucket == bucket_size-1) {
			// One bucket is full, process those that are also half full, in order
			*ppos_bucket = bucket_size;
			typename std::set<idx_bucket_t>::const_iterator it;
			for (it = hfull_buckets.begin(); it != hfull_buckets.end(); it++) {
				idx_bucket_t idx = *it;
				bucket_pos_t bsize = buckets_pos[idx] & MASK_BUCKET_POS;
				red_ref_bucket(bsize, &buckets[idx<<bucket_size_ln2], out);
				buckets_pos[idx] = 0;
			}
			//printf("Done processing half full bucket\n");
			hfull_buckets.clear();
		}
		else {
			pos_bucket++;
			if ((pos_bucket & (mask_half_full | 0x80000000)) == mask_half_full) {
				hfull_buckets.insert(idx_bucket);
				pos_bucket |= 0x80000000;
			}
			*ppos_bucket = pos_bucket;
		}
	}
	for (idx_bucket_t i = 0; i < nbuckets; i++) {
		bucket_pos_t pos_bucket = buckets_pos[i] & MASK_BUCKET_POS;
		if (pos_bucket > 0) {
			red_ref_bucket(pos_bucket, &buckets[i<<bucket_size_ln2], out);
		}
	}
	BENCH_END(red_buckets, "red-buckets-ordered", n, sizeof(uint32_t), 0, sizeof(uint64_t));
}

template <typename bucket_pos_t, typename bucket_v_t, typename idx_bucket_t>
void red_buckets_ordered_noset(const size_t n, const size_t nbits_d, uint_ap in, uint64_ap out, bucket_v_t* buckets, const size_t nbuckets_ln2, const uint8_t bucket_size_ln2)
{
	// Last bit of bucket_pos_t is reserved for knowing if it is half full set
	const uint8_t nbits_shift = (nbits_d+6)-nbuckets_ln2;
	if (nbits_shift > sizeof(bucket_v_t)*8) {
		printf("error: nbits_shift > %lu.\n", sizeof(bucket_v_t)*8);
		return;
	}
	const idx_bucket_t nbuckets = 1<<nbuckets_ln2;
	const bucket_v_t mask_v = (1<<nbits_shift) - 1;
	bucket_pos_t buckets_pos[nbuckets];
	memset(buckets_pos, 0, sizeof(bucket_pos_t)*nbuckets);
	uint32_t v;
	const bucket_pos_t bucket_size = (1<<bucket_size_ln2);
	BENCH_START(red_buckets)
	for (size_t i = 0; i < n; i++) {
		v = in[i];
		const idx_bucket_t idx_bucket = v>>nbits_shift;
		bucket_pos_t* ppos_bucket = &buckets_pos[idx_bucket];
		bucket_pos_t pos_bucket = *ppos_bucket;
		const bucket_pos_t real_pos_bucket = pos_bucket & MASK_BUCKET_POS;
		bucket_v_t* b = &buckets[idx_bucket<<bucket_size_ln2];
		b[real_pos_bucket] = (bucket_v_t) v;
		if (real_pos_bucket == bucket_size-1) {
			// One bucket is full, process those that are also half full, in order
			*ppos_bucket = bucket_size | 0x80000000;
			for (bucket_pos_t idx = 0; idx < nbuckets; idx++) {
				if (buckets_pos[idx] & 0x80000000) {
					bucket_pos_t bsize = buckets_pos[idx] & MASK_BUCKET_POS;
					red_ref_bucket(bsize, &buckets[idx<<bucket_size_ln2], out);
					buckets_pos[idx] = 0;
				}
			}
		}
		else {
			pos_bucket++;
			if (pos_bucket == (nbuckets>>1)) {
				pos_bucket |= 0x80000000;
			}
			*ppos_bucket = pos_bucket;
		}
	}
	for (idx_bucket_t i = 0; i < nbuckets; i++) {
		bucket_pos_t pos_bucket = buckets_pos[i] & MASK_BUCKET_POS;
		if (pos_bucket > 0) {
			red_ref_bucket(pos_bucket, &buckets[i<<bucket_size_ln2], out);
		}
	}
	BENCH_END(red_buckets, "red-buckets-ordered-noset", n, sizeof(uint32_t), 0, sizeof(uint64_t));
}

#if 0
template <typename bucket_pos_t, typename bucket_v_t, typename idx_bucket_t>
void red_buckets2(const size_t n, const size_t nbits_d, uint_ap in, uint64_ap out, bucket_v_t* buckets, const size_t nbuckets_ln2, const uint8_t bucket_size_ln2)
{
	const uint8_t nbits_shift = (nbits_d+6)-nbuckets_ln2;
	if (nbits_shift > sizeof(bucket_v_t)*8) {
		printf("error: nbits_shift > %lu.\n", sizeof(bucket_v_t)*8);
		return;
	}
	const bucket_v_t mask_v = (1<<nbits_shift) - 1;
	const idx_bucket_t nbuckets = 1<<nbuckets_ln2;
	bucket_pos_t buckets_pos[nbuckets];
	memset(buckets_pos, 0, sizeof(bucket_pos_t)*nbuckets);
	uint32_t v;
	const bucket_pos_t bucket_size = (1<<bucket_size_ln2);
	BENCH_START(red_buckets)
	for (size_t i = 0; i < n; i++) {
		v = in[i];
		const idx_bucket_t idx_bucket = v>>nbits_shift;
		bucket_pos_t* ppos_bucket = &buckets_pos[idx_bucket];
		bucket_pos_t pos_bucket = *ppos_bucket;
		bucket_v_t* b = &buckets[idx_bucket<<bucket_size_ln2];
		b[pos_bucket] = (bucket_v_t) v&mask_v;
		if (pos_bucket == bucket_size-1) {
			red_ref_bucket(bucket_size, b, &out[idx_bucket<<(nbits_d-nbuckets_ln2)]);
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
			red_ref_bucket(pos_bucket, &buckets[i<<bucket_size_ln2], &out[i<<(nbits_d-nbuckets_ln2)]);
		}
	}
	BENCH_END(red_buckets, "red-buckets-nocommit", n, sizeof(uint32_t), 0, sizeof(uint64_t));
}
#endif

template <typename bucket_pos_t, typename bucket_v_t, typename idx_bucket_t, void (*F)(size_t, bucket_v_t*, uint64_ap, uint64_ap, const size_t)>
void red_buckets_cached(const size_t n, const size_t nbits_d, uint_ap in, uint64_ap out, bucket_v_t* buckets, const size_t nbuckets_ln2, const uint8_t bucket_size_ln2)
{
	const uint8_t nbits_shift = (nbits_d+6)-nbuckets_ln2;
	if (nbits_shift > sizeof(bucket_v_t)*8) {
		printf("error: nbits_shift > %lu.\n", sizeof(bucket_v_t)*8);
		return;
	}
	const bucket_v_t mask_v = (1<<nbits_shift) - 1;
	const idx_bucket_t nbuckets = 1<<nbuckets_ln2;
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
		b[pos_bucket] = v&mask_v;
		if (pos_bucket == bucket_size-1) {
			F(bucket_size, b, &out[idx_bucket<<(nbits_d-nbuckets_ln2)], out_cache, sout_cache);
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
			F(pos_bucket, &buckets[i<<bucket_size_ln2], &out[i<<(nbits_d-nbuckets_ln2)], out_cache, sout_cache);
		}
	}
	BENCH_END(red_buckets, "red-buckets-stream", n, sizeof(uint32_t), 0, sizeof(uint64_t));
}

template <typename bucket_pos_t, typename bucket_v_t, typename idx_bucket_t, void (*F)(size_t, bucket_v_t*, uint64_ap, uint64_ap, const size_t)>
void red_buckets_cached_omp(size_t n, size_t nbits_d, uint_ap in, size_t nbuckets_ln2, uint8_t bucket_size_ln2, int nth)
{
	uint8_t nbits_shift = (nbits_d+6)-nbuckets_ln2;
	bucket_v_t mask_v = (1<<nbits_shift) - 1;
	idx_bucket_t nbuckets = 1<<nbuckets_ln2;
	bucket_pos_t bucket_size = (1<<bucket_size_ln2);
	size_t sout_cache = (1<<(nbits_d-nbuckets_ln2));
	//uint64_ap outs[nth];
	//BENCH_START(red_buckets)
	double time_taken = 0;
#pragma omp parallel num_threads(nth) firstprivate(nbits_shift,mask_v,nbuckets,bucket_size,sout_cache,n,nbits_d)
	{
		bucket_pos_t buckets_pos[nbuckets];
		memset(buckets_pos, 0, sizeof(bucket_pos_t)*nbuckets);
		uint64_ap out_cache;
		posix_memalign((void**) &out_cache, 16, sout_cache*sizeof(uint64_t));
		bucket_v_t* buckets;
		posix_memalign((void**) &buckets, 16, nbuckets*bucket_size*sizeof(bucket_v_t));
		uint64_ap out;
		posix_memalign((void**) &out, 16, (1<<nbits_d)*sizeof(uint64_t));
		tbb::tick_count start = tbb::tick_count::now();
#pragma omp for
		for (size_t i = 0; i < n; i++) {
			uint32_t v;
			v = in[i];
			const uint32_t idx_bucket = v>>nbits_shift;
			bucket_pos_t* ppos_bucket = &buckets_pos[idx_bucket];
			bucket_pos_t pos_bucket = *ppos_bucket;
			bucket_v_t* b = &buckets[idx_bucket<<bucket_size_ln2];
			b[pos_bucket] = v&mask_v;
			if (pos_bucket == bucket_size-1) {
				F(bucket_size, b, &out[idx_bucket<<(nbits_d-nbuckets_ln2)], out_cache, sout_cache);
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
				F(pos_bucket, &buckets[i<<bucket_size_ln2], &out[i<<(nbits_d-nbuckets_ln2)], out_cache, sout_cache);
			}
		}
		tbb::tick_count end = tbb::tick_count::now();
#pragma omp critical
		{
			time_taken = std::max(time_taken, (end-start).seconds());
		}
	}
	printf("%d threads: %0.4f ms\n", nth, (time_taken*1000.0));
	//BENCH_END(red_buckets, "red-buckets-omp", n, sizeof(uint32_t), 0, sizeof(uint64_t));
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

	// Bootstrap openmp
	int a = 0;
#pragma omp parallel for
	for (int i = 0; i < 1000000; i++) {
		a += i;
	}

	size_t d = atoll(argv[1]);
	size_t n = atoll(argv[2]);
	uint16_t nbuckets_ln2 = atoi(argv[3]);
	uint8_t bucket_size_ln2 = atoi(argv[4]);
	uint32_t bucket_size = 1<<bucket_size_ln2;
	uint16_t nbuckets = 1<<(nbuckets_ln2);

	uint32_t nbits_d = (uint32_t) (log2f((double)d));
	//printf("nbits_d: %d\n", nbits_d);
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
	//printf("Buckets size: %f KB\n", ((double)sbuckets)/1024.0);
	posix_memalign((void**) &buckets, 16, sbuckets);
	memset(buckets, 0, sbuckets);
	
	BENCH_START(red_ref);
	red_ref(n, d, in, out_ref);
	BENCH_END(red_ref, "red-ref", n, sizeof(uint32_t), d, sizeof(uint64_t));


	memset(out, 0, d*sizeof(uint64_t));
#if 0
	for (int i = 1; i <= omp_get_max_threads(); i++) {
		if (nbuckets_ln2 >= 9) {
			//red_buckets_nocommit<uint16_t, uint16_t, uint16_t>(n, nbits_d, in, out, (uint16_t*) buckets, nbuckets_ln2, bucket_size_ln2);
			//red_buckets_cached<uint32_t, uint16_t, uint16_t, red_bucket_commit_clear<uint16_t> >(n, nbits_d, in, out, (uint16_t*) buckets, nbuckets_ln2, bucket_size_ln2);
			red_buckets_cached_omp<uint32_t, uint16_t, uint16_t, red_bucket_nocommit_noclear<uint16_t> >(n, nbits_d, in, nbuckets_ln2, bucket_size_ln2, i);
		}
		else {
			//red_buckets_nocommit<uint16_t, uint32_t, uint16_t>(n, nbits_d, in, out, buckets, nbuckets_ln2, bucket_size_ln2);
			//red_buckets_cached<uint32_t, uint32_t, uint16_t, red_bucket_commit_clear<uint32_t> >(n, nbits_d, in, out, buckets, nbuckets_ln2, bucket_size_ln2);
			red_buckets_cached_omp<uint32_t, uint32_t, uint16_t, red_bucket_nocommit_noclear<uint32_t> >(n, nbits_d, in, nbuckets_ln2, bucket_size_ln2, i);
		}
	}
	//CHECK(memcmp(out_ref, out, d*sizeof(uint64_t)) == 0);
#endif

	//if (bucket_size_ln2 < 16)
	//	red_buckets<uint16_t, uint32_t>(n, nbits_d, in, out, buckets, nbuckets_ln2, bucket_size_ln2);
	memset(out, 0, d*sizeof(uint64_t));
	red_buckets<uint32_t, uint32_t, uint16_t>(n, nbits_d, in, out, buckets, nbuckets_ln2, bucket_size_ln2);
	CHECK(memcmp(out_ref, out, d*sizeof(uint64_t)) == 0);
	memset(out, 0, d*sizeof(uint64_t));
	red_buckets_ordered<uint32_t, uint32_t, uint16_t>(n, nbits_d, in, out, buckets, nbuckets_ln2, bucket_size_ln2);
	CHECK(memcmp(out_ref, out, d*sizeof(uint64_t)) == 0);
	memset(out, 0, d*sizeof(uint64_t));
	red_buckets_ordered_noset<uint32_t, uint32_t, uint16_t>(n, nbits_d, in, out, buckets, nbuckets_ln2, bucket_size_ln2);
	CHECK(memcmp(out_ref, out, d*sizeof(uint64_t)) == 0);

	if (nbuckets_ln2 >= 9) {
		printf("Using 16-bit for the bucket buffers...\n");
		/*memset(out, 0, d*sizeof(uint64_t));
		red_buckets<uint32_t, uint16_t, uint16_t>(n, nbits_d, in, out, (uint16_t*) buckets, nbuckets_ln2, bucket_size_ln2);
		CHECK(memcmp(out_ref, out, d*sizeof(uint64_t)) == 0);
		memset(out, 0, d*sizeof(uint64_t));
		red_buckets_ordered<uint32_t, uint16_t, uint16_t>(n, nbits_d, in, out, (uint16_t*) buckets, nbuckets_ln2, bucket_size_ln2);
		CHECK(memcmp(out_ref, out, d*sizeof(uint64_t)) == 0);
		memset(out, 0, d*sizeof(uint64_t));
		red_buckets_ordered_noset<uint32_t, uint16_t, uint16_t>(n, nbits_d, in, out, (uint16_t*) buckets, nbuckets_ln2, bucket_size_ln2);
		CHECK(memcmp(out_ref, out, d*sizeof(uint64_t)) == 0);*/
	}

	uint8_ap out8;
	posix_memalign((void**) &out8, 16, sizeof(uint8_t)*(d*8*8));
	BENCH_START(red_ref8);
	red_ref_8(n, d, in, out8);
	BENCH_END(red_ref8, "red-ref8", n, sizeof(uint32_t), d*8*8, sizeof(uint8_t));
	

	tbb::parallel_sort(in, in+n);
	BENCH_START(red_sort);
	red_ref(n, d, in, out_ref);
	BENCH_END(red_sort, "red-after-sort", n, sizeof(uint32_t), d, sizeof(uint64_t));

	return 0;
}
