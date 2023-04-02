//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <iostream>
#include <pvkernel/core/PVSelBitField.h>

#include <pvkernel/core/inendi_assert.h>

static constexpr size_t SELECTION_COUNT = 100000000;

template <class A, class B>
bool show_diff(A const& cmp, B const& ref)
{
	size_t size_ref = ref.size();
	size_t size_cmp = cmp.size();
	bool ret = false;
	if (size_ref != size_cmp) {
		std::cerr << "Size differs: cmp " << size_cmp << " != ref " << size_ref << std::endl;
		ret = true;
	}
	for (size_t i = 0; i < std::min(size_cmp, size_ref); i++) {
		if (cmp[i] != ref[i]) {
			std::cerr << i << ": cmp " << cmp[i] << " != ref " << ref[i] << std::endl;
			ret = true;
		}
	}

	return ret;
}

template <class F>
void basic_visit(PVCore::PVSelBitField const& bits, F const& f, PVRow const b, PVRow const a)
{
	for (PVRow r = a; r < b; r++) {
		if (bits.get_line(r)) {
			f(r);
		}
	}
}

void do_tests(PVCore::PVSelBitField const& bits, std::vector<std::pair<PVRow, PVRow>> const& ranges)
{
	std::vector<PVRow> ref;
	std::vector<PVRow> cur;

	for (std::pair<PVRow, PVRow> const& r : ranges) {
		const PVRow a = r.first;
		const PVRow b = r.second;
		assert(b > a);
		const PVRow nrows = b - a;
		ref.reserve(nrows);
		cur.reserve(nrows);

		std::cout << "Testing with range [" << a << "," << b << "[..." << std::endl;

		ref.clear();
		basic_visit(bits, [&ref](PVRow const r) { ref.push_back(r); }, b, a);

		cur.clear();
		bits.visit_selected_lines([&cur](PVRow const r) { cur.push_back(r); }, b, a);
		if (show_diff(cur, ref)) {
			std::cerr << "visit_selected_lines failed" << std::endl;
			exit(1);
		}
	}
}

void do_tests_packet(PVCore::PVSelBitField& bits)
{
	for (uint32_t i = 0; i < 16; ++i) {
		uint32_t v =
		    (i << 28) | (i << 24) | (i << 20) | (i << 16) | (i << 12) | (i << 8) | (i << 4) | (i);
		memset(bits.get_buffer(), v, 2048);
		for (uint32_t j = 0; j < 32; ++j) {
			for (uint32_t k = 0; k < 64; k += 4) {
				PV_VALID(bits.get_lines_fast((j * 32) + k, 4), i, "i", i, "j", j, "k", k);
			}
		}
	}
}

int main()
{
	srand(time(nullptr));

	PVCore::PVSelBitField bits(SELECTION_COUNT);

	std::vector<std::pair<PVRow, PVRow>> ranges;
	// Ranges are semi-open ([X, Y[)
	// Only 1 chunk (64-bits per chunk) involved
	ranges.emplace_back(0, 5);
	ranges.emplace_back(0, 11);
	ranges.emplace_back(7, 10);
	ranges.emplace_back(8, 18);
	// 2 chunks (64-bits per chunk) involved
	ranges.emplace_back(63, 67);
	ranges.emplace_back(60, 67);
	// 3 chunks (64-bits per chunk) involved
	ranges.emplace_back(60, 140);
	ranges.emplace_back(63, 150);
	// 4 chunks (64-bits per chunk) involved
	ranges.emplace_back(60, 205);
	ranges.emplace_back(63, 215);
#ifdef TESTS_LONG
	// More than 4 chunks involved
	ranges.push_back(std::make_pair(65, std::min(1000008, SELECTION_COUNT)));
	ranges.push_back(std::make_pair(111, std::min(1000008, SELECTION_COUNT)));
	ranges.push_back(std::make_pair(0, std::min(1500000, SELECTION_COUNT)));
	// Prime-number ranges
	ranges.push_back(std::make_pair(89767, std::min(99991, SELECTION_COUNT)));
	// Previous bugous ranges
	//
	// There was an issue with the epilogue if the last chunk was indeed full
	ranges.push_back(std::make_pair(57262, std::min(58624, SELECTION_COUNT)));
	// Related to #245
	ranges.push_back(std::make_pair(81990, std::min(102886, SELECTION_COUNT)));
#endif

	std::cout << "Tests with full selection..." << std::endl;
	bits.select_all();
	do_tests(bits, ranges);

	std::cout << "Tests with empty selection..." << std::endl;
	bits.select_none();
	do_tests(bits, ranges);

	std::cout << "Tests with \"even\" selection..." << std::endl;
	bits.select_even();
	do_tests(bits, ranges);

	std::cout << "Tests with \"odd\" selection..." << std::endl;
	bits.select_odd();
	do_tests(bits, ranges);

	std::cout << "Tests with random selection (1)..." << std::endl;
	bits.select_random();
	do_tests(bits, ranges);

	std::cout << "Tests with random selection (2)..." << std::endl;
	bits.select_random();
	do_tests(bits, ranges);

	std::cout << "Tests with 88104, 88106, 88111, 88113 (related to #245)..." << std::endl;
	bits.select_none();
	bits.set_bit_fast(88104);
	bits.set_bit_fast(88106);
	bits.set_bit_fast(88111);
	bits.set_bit_fast(88113);
	do_tests(bits, ranges);

	std::cout << "Tests with 0, 1, 2, 3, 4..." << std::endl;
	bits.select_none();
	bits.set_bit_fast(0);
	bits.set_bit_fast(1);
	bits.set_bit_fast(2);
	bits.set_bit_fast(3);
	bits.set_bit_fast(4);
	do_tests(bits, ranges);

	std::cout << "Tests with the first chunk full and a random second-one..." << std::endl;
	bits.select_none();
	for (size_t i = 0; i < PVCore::PVSelBitField::CHUNK_SIZE; i++) {
		bits.set_bit_fast(i);
	}
	for (size_t i = PVCore::PVSelBitField::CHUNK_SIZE; i < PVCore::PVSelBitField::CHUNK_SIZE * 2;
	     i++) {
		if (rand() & 1) {
			bits.set_bit_fast(i);
		}
	}
	do_tests(bits, ranges);

	std::cout << "Tests with the two first chunks full and a random third one..." << std::endl;
	bits.select_none();
	for (size_t i = 0; i < PVCore::PVSelBitField::CHUNK_SIZE * 2; i++) {
		bits.set_bit_fast(i);
	}
	for (size_t i = PVCore::PVSelBitField::CHUNK_SIZE * 2;
	     i < PVCore::PVSelBitField::CHUNK_SIZE * 3; i++) {
		if (rand() & 1) {
			bits.set_bit_fast(i);
		}
	}
	do_tests(bits, ranges);

	for (size_t i = 1; i <= 128; i++) {
		std::cout << "Tests with " << i << " randomly selected lines" << std::endl;
		bits.select_random(i);
		do_tests(bits, ranges);
	}

	std::cout << "Tests with 10000 randomly selected lines" << std::endl;
	bits.select_random(10000);
	do_tests(bits, ranges);

	std::cout << "Tests of get_lines_fast..." << std::endl;
	do_tests_packet(bits);

	return 0;
}
