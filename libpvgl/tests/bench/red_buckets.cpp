#include <common/bench.h>
#include <common/common.h>

#include <iostream>

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include <pvkernel/core/picviz_intrin.h>

typedef uint32_t DECLARE_ALIGN(16) * uint_ap;
typedef uint64_t DECLARE_ALIGN(16) * uint64_ap;

uint32_t _and_mask_d = 0;

void red_ref(size_t n, size_t d, uint_ap in, uint64_ap out)
{
	uint32_t v;
	for (size_t i = 0; i < n; i++) {
		v = in[i];
		out[(v>>6)&((1<<11)-1)] |= 1LL<<(v&63);
	}
}

void red_buckets(const size_t n, const size_t d, uint_ap in, uint64_ap out, uint_ap* buckets, const size_t nbuckets, const size_t nbuckets_ln2, const size_t bucket_size)
{
	const uint8_t nbits_shift = 25-nbuckets_ln2;
	uint32_t buckets_pos[nbuckets];
	memset(buckets_pos, 0, sizeof(uint32_t)*nbuckets);
	for (size_t i = 0; i < n; i++) {
		uint32_t v = in[i];
		uint32_t idx_bucket = v>>nbits_shift;
		uint32_t pos_bucket = buckets_pos[idx_bucket];
		buckets[idx_bucket][pos_bucket] = v;
		if (pos_bucket == bucket_size-1) {
			red_ref(bucket_size, d, buckets[idx_bucket], out);
			pos_bucket = 0;
		}
		else {
			pos_bucket++;
		}
		buckets_pos[idx_bucket] = pos_bucket;
	}
	for (size_t i = 0; i < nbuckets; i++) {
		uint32_t pos_bucket = buckets_pos[i];
		if (pos_bucket > 0) {
			red_ref(pos_bucket, d, buckets[i], out);
		}
	}
}

void init_rand_int(size_t n, uint_ap buf)
{
	for (size_t i = 0; i < n; i++) {
		buf[i] = (((rand()<<1)+1)&(_and_mask_d<<6)) | ((rand())&63);
	}
}

int main(int argc, char** argv)
{
	if (argc < 5) {
		std::cerr << "Usage: " << argv[0] << " dint nint nbuckets_log2 size_bucket" << std::endl;
		return 1;
	}

	size_t d = atoll(argv[1]);
	size_t n = atoll(argv[2]);
	size_t nbuckets_ln2 = atoll(argv[3]);
	size_t bucket_size = atoll(argv[4]);
	size_t nbuckets = 1<<(nbuckets_ln2);

	printf("%f\n", log2f((double)d));
	_and_mask_d = ((1<<((uint64_t) (log2f((double)d)))) - 1);

	srand(time(NULL));

	uint_ap in; uint64_ap out;
	posix_memalign((void**) &in, 16, n*sizeof(uint32_t));
	posix_memalign((void**) &out, 16, d*sizeof(uint64_t));
	init_rand_int(n, in);

	uint32_t* buckets[nbuckets];
	for (size_t i = 0; i < nbuckets; i++) {
		posix_memalign((void**) &buckets[i], 16, bucket_size*sizeof(uint32_t));
		memset(buckets[i], 0, bucket_size*sizeof(uint32_t));
	}

	/*
	BENCH_START(red_ref);
	red_ref(n, d, in, out);
	BENCH_END(red_ref, "red-ref", n, sizeof(uint32_t), d, sizeof(uint64_t));
	*/

	BENCH_START(red_buckets)
	red_buckets(n, d, in, out, buckets, nbuckets, nbuckets_ln2, bucket_size);
	BENCH_END(red_buckets, "red-buckets", n, sizeof(uint32_t), d, sizeof(uint64_t));

	return 0;
}
