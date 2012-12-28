/**
 * \file visit_bits.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/picviz_assert.h>
#include <pvkernel/core/picviz_bench.h>
#include <pvkernel/core/picviz_intrin.h>
#include <pvkernel/core/PVBitCount.h>
#include <pvkernel/core/PVBitVisitor.h>
#include <pvkernel/core/PVByteVisitor.h>
#include <iostream>
#include <assert.h>
#include <time.h>

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

#define MAX_SLICE_SIZE 1024
size_t get_slice_size(size_t i)
{
	return rand()%MAX_SLICE_SIZE + 1;
}

int main()
{
	uint64_t v = (1UL<<0) | (1UL<<4) | (1UL<<11) | (1UL<<18) | (1UL<<26) | (1UL<<35) | (1UL << 46) | (1UL<<59);
	PVCore::PVBitVisitor::visit_bits(v, [=](unsigned int b) { std::cout << b << std::endl; });

	uint64_t* buf;
	posix_memalign((void**) &buf, 16, sizeof(uint64_t)*SIZE_BUF);
	memset(buf, 0xA0, sizeof(uint64_t)*SIZE_BUF);

	srand(time(NULL));
#if 0
	BENCH_START(sse4);
	uint32_t ret1 = count_bits_sse(SIZE_BUF, buf);
	BENCH_STOP(sse4);
	PV_STAT_PROCESS_BW("count_bits_sse", (sizeof(uint64_t) * SIZE_BUF) / BENCH_END_TIME(sse4));

	BENCH_START(vec);
	uint32_t ret0 = count_bits(SIZE_BUF, buf);
	BENCH_STOP(vec);
	PV_STAT_PROCESS_BW("count_bits_vec", (sizeof(uint64_t) * SIZE_BUF) / BENCH_END_TIME(vec));

	BENCH_START(lib);
	uint32_t ret2 = PVCore::PVBitCount::bit_count(SIZE_BUF, buf);
	BENCH_STOP(lib);
	PV_STAT_PROCESS_BW("count_bits_lib", (sizeof(uint64_t) * SIZE_BUF) / BENCH_END_TIME(lib));

	PV_ASSERT_VALID(ret0 == ret1, "ret0", ret0, "ret1", ret1);
	PV_ASSERT_VALID(ret0 == ret2, "ret0", ret0, "ret2", ret2);
	PV_ASSERT_VALID(ret1 == ret2, "ret1", ret1, "ret2", ret2);
#endif

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

	std::cout << "visit_nth_slices:" << std::endl;
	const char* str_test = "a\0b\0c\0d\0e\0f\0g\0h\0i\0j\0k\0l\0m\0n\0o\0p\0q\0r\0s\0t\0u\0v\0w\0x\0y\0z\0aa\0bb\0cc\0dd\0ee\0ff\0gg\0hh\0ii\0jj\0kk\0ll\0mm\0nn";
	for (int i = 0; i < 40; i++) {
		PVCore::PVByteVisitor::visit_nth_slice((const uint8_t*) str_test, 132, i, [=](uint8_t const* str, size_t n)
			{
				// fwrite(str, 1, n, stdout);
				// printf("\n");
			});
	}

	free(buf);

#define NSLICE 1024000
	char* str2 = (char*) malloc(NSLICE*(MAX_SLICE_SIZE+1));
	char* cur_str2 = str2;
	for (size_t i = 0; i < NSLICE; i++) {
		size_t sslice = get_slice_size(i);
		memset(cur_str2, 'a' + (i%26), sslice);
		cur_str2[sslice] = '\0';
		cur_str2 += sslice+1;
	}
	size_t str2_size = (uintptr_t)cur_str2-(uintptr_t)str2;

	BENCH_START(slice);
	PVCore::PVByteVisitor::visit_nth_slice((const uint8_t*) str2, str2_size, NSLICE-1, [=](uint8_t const* /*str*/, size_t /*n*/) { });
#if 0
	for (size_t i = 0; i < NSLICE; i++) {
		PVCore::PVByteVisitor::visit_nth_slice((const uint8_t*) str2, str2_size, i, [=](uint8_t const* str, size_t n)
				{
					PV_ASSERT_VALID(n == get_slice_size(i));
					char buf[get_slice_size(i)];
					memset(&buf[0], 'a' + i%26, get_slice_size(i));
					PV_ASSERT_VALID(memcmp(str, buf, n) == 0);
					//fwrite(str, 1, n, stdout);
					//printf("\n");
				});
	}
#endif
	//BENCH_END(slice, "slice", sizeof(char), str2_size, 1, 1);
	BENCH_STOP(slice);
	free(str2);

	return 0;
}
