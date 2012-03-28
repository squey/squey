#include <common/common.h>
#include <common/bench.h>
#include <iostream>

#include <cstdlib>
#include <ctime>

#define COUNT_BITS_UINT32(v,res) \
	v = v - ((v >> 1) & 0x55555555);\
	v = (v & 0x33333333) + ((v >> 2) & 0x33333333);\
	res += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;

static inline uint32_t count_bits_naive(size_t n, const uint32_t* DECLARE_ALIGN(16) data)
{
	uint32_t ret = 0;
	for (size_t i = 0; i < n; i++) {
		uint32_t v = data[i];
		for (uint32_t j = 0; j <= 31; j++) {
			if ((v & (1 << j)) != 0) {
				ret++;
			}
		}
	}
	return ret;
}

static inline uint32_t count_bits(size_t n, const uint32_t* DECLARE_ALIGN(16) data)
{
	uint32_t ret = 0;
	for (size_t i = 0; i < n; i++) {
		uint32_t v = data[i];
		COUNT_BITS_UINT32(v,ret);
	}
	return ret;
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " n" <<std::endl;
		return 1;
	}

	int a = 0;
#pragma omp parallel for
	for (int i = 0; i < 10000000; i++) {
		a += i;
	}

	srand(time(NULL));

	const size_t n = atoll(argv[1]);
	uint32_t* DECLARE_ALIGN(16) buf = (uint32_t*) malloc(sizeof(uint32_t)*n);
	for (size_t i = 0; i < n; i++) {
		buf[i] = rand();
	}

	size_t ref;
	BENCH_START(count);
	ref = count_bits_naive(n, buf);
	BENCH_END(count, "ref", n, sizeof(uint32_t), 1, sizeof(uint32_t));

	BENCH_START(count2);
	size_t cmp = count_bits(n, buf);
	BENCH_END(count2, "op1", n, sizeof(uint32_t), 1, sizeof(uint32_t));
	CHECK(cmp == ref);

	return 0;
}
