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

void check_bit_visitor(int *bits_ref, size_t size)
{
	uint64_t v = 0;
	for (size_t i = 0; i < size; i++) {
		v |= 1ULL<<(bits_ref[i]);
	}

	std::vector<int> bits;
	PVCore::PVBitVisitor::visit_bits(v, [&](unsigned int b) { bits.push_back(b); });

	PV_ASSERT_VALID(size == bits.size());
	PV_ASSERT_VALID(std::equal(bits.begin(), bits.end(), bits_ref));
}

void check_slices_visitor(std::vector<std::string> const& slices)
{
	size_t buf_size = 0;
	for (std::string const& s: slices) {
		buf_size += s.size() + 1;
	}
	char* buf = (char*) malloc(buf_size);
	char* cur_buf = buf;
	for (std::string const& s: slices) {
		memcpy(cur_buf, s.c_str(), s.size()+1);
		cur_buf += s.size()+1;
	}

	std::vector<std::string> slices_ret;
	slices_ret.reserve(slices.size());
	for (size_t i = 0; i < slices.size(); i++) {
		PVCore::PVByteVisitor::visit_nth_slice((const uint8_t*) buf, buf_size, i, [&slices_ret](uint8_t const* str, size_t n)
				{
					slices_ret.push_back(std::string((const char*) str, n));
				});
	}

	PV_ASSERT_VALID(slices == slices_ret);
}

void generate_random_slices(std::vector<std::string>& slices, size_t n, size_t min_len, size_t max_len)
{
	slices.clear();
	slices.reserve(n);
	// range is [min_len,max_len]
	for (size_t i = 0; i < n; i++) {
		size_t str_size = (rand() % (max_len + 1 - min_len)) + min_len;

		std::string rand_str;
		rand_str.reserve(str_size);
		for (size_t c = 0; c < str_size; c++) {
			rand_str.push_back((rand() % 26) + 'a');
		}
		slices.push_back(std::move(rand_str));
	}
}

int main()
{
	{
		int bits[] = {0, 4, 11, 18, 26, 35, 46, 59};
		check_bit_visitor(bits, sizeof(bits)/sizeof(int));
	}

	{
		int bits[64];
		for (int i = 0; i < 64; i++) {
			bits[i] = i;
		}
		check_bit_visitor(bits, sizeof(bits)/sizeof(int));
	}

	//__m128i v_sse = _mm_set1_epi32(0x00FF0000);
	//PVCore::PVByteVisitor::visit_bytes(v_sse, [=](size_t b) { std::cout << b << std::endl; });

	// Slices visitor
	std::vector<std::string> slices;
	for (uint8_t i = 0; i < 26; i++) {
		char c = 'a' + (char)i;
		slices.push_back(std::string(&c, 1));
	}
	check_slices_visitor(slices);

	slices.clear();
	for (uint8_t i = 0; i < 26; i++) {
		char c = 'a' + (char)i;
		slices.push_back(std::string(i, c));
	}
	check_slices_visitor(slices);

	srand(time(NULL));

#define NSLICES 2048

	generate_random_slices(slices, NSLICES, 1, 2048);
	check_slices_visitor(slices);

	generate_random_slices(slices, NSLICES*20, 1, 15);
	check_slices_visitor(slices);

	generate_random_slices(slices, NSLICES*10, 1, 4);
	check_slices_visitor(slices);

	return 0;
}
