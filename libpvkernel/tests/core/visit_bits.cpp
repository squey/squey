/**
 * \file visit_bits.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/picviz_intrin.h>
#include <pvkernel/core/picviz_bench.h>
#include <pvkernel/core/PVBitCount.h>
#include <pvkernel/core/PVBitVisitor.h>
#include <pvkernel/core/PVByteVisitor.h>
#include <iostream>

#define COUNT_BITS_UINT64(ret,v)\
	v = v - ((v >> 1) & 0x5555555555555555ULL);\
	v = (v & 0x3333333333333333ULL) + ((v >> 2) & 0x3333333333333333ULL);\
	ret += (((v + (v >> 4)) & 0x0f0f0f0f0f0f0f0f) * 0x0101010101010101) >> 56;\

static uint32_t count_bits(size_t n, const uint64_t* data)
{
	uint32_t ret = 0;
	for (size_t i = 0; i < n; i++) {
		uint64_t v = data[i];
		COUNT_BITS_UINT64(ret,v);
	}   
	return ret;
}

static uint32_t count_bits_sse(size_t n, const uint64_t* data)
{
	uint32_t ret = 0;
	for (size_t i = 0; i < n; i++) {
		const uint64_t v = data[i];
		ret += _mm_popcnt_u64(v);
	}   
	return ret;
}

#define SIZE_BUF ((1UL<<(32-6))-1)

int main()
{
	uint64_t v = (1UL<<0) | (1UL<<4) | (1UL<<11) | (1UL<<18) | (1UL<<26) | (1UL<<35) | (1UL << 46) | (1UL<<59);
	PVCore::PVBitVisitor::visit_bits(v, [=](unsigned int b) { std::cout << b << std::endl; });

	uint64_t* buf;
	posix_memalign((void**) &buf, 16, sizeof(uint64_t)*SIZE_BUF);
	memset(buf, 0xA0, sizeof(uint64_t)*SIZE_BUF);

	BENCH_START(sse4);
	uint32_t ret1 = count_bits_sse(SIZE_BUF, buf);
	BENCH_END(sse4, "sse4", sizeof(uint64_t), SIZE_BUF, sizeof(uint32_t), 1);

	BENCH_START(vec);
	uint32_t ret0 = count_bits(SIZE_BUF, buf);
	BENCH_END(vec, "vec", sizeof(uint64_t), SIZE_BUF, sizeof(uint32_t), 1);

	BENCH_START(lib);
	uint32_t ret2 = PVCore::PVBitCount::bit_count(SIZE_BUF, buf);
	BENCH_END(lib, "lib", sizeof(uint64_t), SIZE_BUF, sizeof(uint32_t), 1);

	v = (0xFFULL<<(2*8)) | (0xFFULL<<(4*8)) | (0xFFULL<<(5*8)) | (0xFFULL<<(7*8));
	PVCore::PVByteVisitor::visit_bytes(v, [=](size_t b) { std::cout << b << std::endl; });

	std::cout << std::endl;

	v = 0xFFULL;
	PVCore::PVByteVisitor::visit_bytes(v, [=](size_t b) { std::cout << b << std::endl; });

	std::cout << std::endl;
	v = 0xFFFFFFFFFFFFFFFFULL;
	PVCore::PVByteVisitor::visit_bytes(v, [=](size_t b) { std::cout << b << std::endl; });

	std::cout << std::endl;
	__m128i v_sse = _mm_set1_epi32(0x00FF0000);
	PVCore::PVByteVisitor::visit_bytes(v_sse, [=](size_t b) { std::cout << b << std::endl; });

	printf("%u %u %u\n", ret0, ret1, ret2);

	free(buf);

	return 0;
}
