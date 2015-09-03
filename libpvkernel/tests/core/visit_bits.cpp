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

	// Create an aligned and an un-aligned buffer
	char *buf, *abuf, *abuf_org;
	posix_memalign((void**) &buf, 16, buf_size*sizeof(char));
	posix_memalign((void**) &abuf_org, 16, buf_size*sizeof(char) + 5);
	abuf = abuf_org + 5;
	char* cur_buf = buf;
	char* cur_abuf = abuf;
	for (std::string const& s: slices) {
		memcpy(cur_buf, s.c_str(), s.size()+1);
		memcpy(cur_abuf, s.c_str(), s.size()+1);
		cur_buf += s.size()+1;
		cur_abuf += s.size()+1;
	}

	std::vector<std::string> slices_ret;
	std::vector<std::string> slices_aret;
	slices_ret.reserve(slices.size());
	slices_aret.reserve(slices.size());
	for (size_t i = 0; i < slices.size(); i++) {
		PVCore::PVByteVisitor::visit_nth_slice((const uint8_t*) buf, buf_size, i, [&slices_ret](uint8_t const* str, size_t n)
				{
					slices_ret.push_back(std::string((const char*) str, n));
				});
		PVCore::PVByteVisitor::visit_nth_slice((const uint8_t*) abuf, buf_size, i, [&slices_aret](uint8_t const* str, size_t n)
				{
					slices_aret.push_back(std::string((const char*) str, n));
				});
	}

	free(buf);
	free(abuf_org);

	PV_ASSERT_VALID(slices == slices_ret);
	PV_ASSERT_VALID(slices == slices_aret);
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
	std::cout << "Check bit visitor" << std::endl;
	{
		int bits[64];
		for (int i = 0; i < 64; i++) {
			bits[i] = i;
		}
		check_bit_visitor(bits, sizeof(bits)/sizeof(int));
	}
	std::cout << "done" << std::endl;

	//__m128i v_sse = _mm_set1_epi32(0x00FF0000);
	//PVCore::PVByteVisitor::visit_bytes(v_sse, [=](size_t b) { std::cout << b << std::endl; });

	std::cout << "Check slice visitor" << std::endl;
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
	std::cout << "done" << std::endl;

	srand(time(NULL));

#define NSLICES 2048

	std::cout << "Check slice visitor with random value" << std::endl;
	generate_random_slices(slices, NSLICES, 1, 2048);
	check_slices_visitor(slices);
	std::cout << "done" << std::endl;

#ifdef TESTS_LONG
	generate_random_slices(slices, NSLICES*20, 1, 15);
	check_slices_visitor(slices);

	generate_random_slices(slices, NSLICES*10, 1, 4);
	check_slices_visitor(slices);
#endif

	return 0;
}
