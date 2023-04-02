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

#include <cstdlib>
#include <pvkernel/core/inendi_assert.h>
#include <pvkernel/core/inendi_bench.h>

#include <inendi/PVSelection.h>
#include <iostream>

static constexpr size_t SELECTION_COUNT = 1000000;

int main(void)
{

	// #include "test-env.h"

	Inendi::PVSelection* selection;
	Inendi::PVSelection* selection2 = new Inendi::PVSelection(SELECTION_COUNT);
	Inendi::PVSelection* selection3 = new Inendi::PVSelection(SELECTION_COUNT);
	Inendi::PVSelection a(SELECTION_COUNT);
	Inendi::PVSelection b(SELECTION_COUNT);

	size_t count;

	/**********************************************************************
	 *
	 * We test creation and deletion of a inendi_selection_t
	 *
	 **********************************************************************/

	selection = new Inendi::PVSelection(SELECTION_COUNT);
	delete (selection);

	a.select_none();
	BENCH_START(sel);
	std::cout << "get_last_nonzero_chunk_index on empty sel: " << a.get_last_nonzero_chunk_index()
	          << std::endl;
	BENCH_END(sel, "bench", sizeof(uint32_t), a.chunk_count(), 1, 1);

	a.select_all();
	BENCH_START(self);
	std::cout << "get_last_nonzero_chunk_index on full sel (should be " << (a.chunk_count() - 1)
	          << "): " << a.get_last_nonzero_chunk_index() << std::endl;
	BENCH_END(self, "bench", sizeof(uint32_t), a.chunk_count(), 1, 1);

	a.select_none();
	a.set_bit_fast(4);
	std::cout << "get_last_nonzero_chunk_index with bit 4: " << a.get_last_nonzero_chunk_index()
	          << std::endl;

	a.set_bit_fast(44);
	std::cout << "get_last_nonzero_chunk_index with bit 44: " << a.get_last_nonzero_chunk_index()
	          << std::endl;

	a.set_bit_fast(32 * 4);
	std::cout << "get_last_nonzero_chunk_index with bit 32*4: " << a.get_last_nonzero_chunk_index()
	          << std::endl;

	a.visit_selected_lines([&](PVRow r) { std::cout << r << ","; });
	std::cout << std::endl;

	a.select_none();
#define NLINES_TEST 10000
	for (int i = 0; i < NLINES_TEST; i++) {
		a.set_bit_fast(rand() % (SELECTION_COUNT / 4));
	}
	std::vector<PVRow> ref, test;
	ref.reserve(NLINES_TEST);
	test.reserve(NLINES_TEST);
	BENCH_START(ref);
	a.visit_selected_lines([&](PVRow r) { ref.push_back(r); });
	BENCH_END(ref, "visit ref", sizeof(uint32_t), a.chunk_count(), sizeof(PVRow), ref.size());

	std::cout << "Visit sse test: " << (ref == test) << std::endl;

	/**********************************************************************
	 *
	 * We test all Generic functions
	 *
	 **********************************************************************/

	std::cout << "we test bit_count() and select_all()\n";

	selection = new Inendi::PVSelection(SELECTION_COUNT);
	selection->select_all();
	count = selection->bit_count();
	PV_VALID(count, SELECTION_COUNT);

	std::cout << "we test select_even()\n";
	selection->select_even();
	count = selection->bit_count();
	PV_VALID(count, SELECTION_COUNT / 2);

	std::cout << "we test select_odd()\n";
	selection->select_odd();
	count = selection->bit_count();
	PV_VALID(count, SELECTION_COUNT / 2);

	// Test of is_empty
	for (int i = 0; i < 256; i++) {
		a.select_none();
		a.set_bit_fast(i);
		PV_ASSERT_VALID(not a.is_empty());
	}

	// Test of C++0x features
	a.select_even();
	b.select_odd();

	Inendi::PVSelection c = a & b;
	PVLOG_INFO("a: %p , b = %p , c = %p\n", &a, &b, &c);
	std::cout << "PVSelection should be empty: PVSelection::is_empty() = " << c.is_empty()
	          << std::endl;

	c = a;
	BENCH_START(original_or);
	c |= b;
	BENCH_END_TRANSFORM(original_or, "Original OR", sizeof(uint32_t), c.chunk_count());
	std::cout << "Last chunk should be " << c.chunk_count() - 1 << " : "
	          << c.get_last_nonzero_chunk_index() << std::endl;

	c = a;
	BENCH_START(opt_or);
	c |= b;
	BENCH_END_TRANSFORM(opt_or, "Opt OR", sizeof(uint32_t), c.chunk_count());
	std::cout << "Last chunk should be " << c.chunk_count() - 1 << " : "
	          << c.get_last_nonzero_chunk_index() << std::endl;

	b.select_none();

	c = a;
	BENCH_START(original_or2);
	c |= b;
	BENCH_END_TRANSFORM(original_or2, "Original OR", sizeof(uint32_t), c.chunk_count());
	std::cout << "Last chunk : " << c.get_last_nonzero_chunk_index() << std::endl;

	c = a;
	BENCH_START(opt_or2);
	c |= b;
	BENCH_END_TRANSFORM(opt_or2, "Opt OR", sizeof(uint32_t), c.chunk_count());
	std::cout << "Last chunk : " << c.get_last_nonzero_chunk_index() << std::endl;

	a.select_none();
	a.set_bit_fast(1);
	ref.clear();
	a.visit_selected_lines([&](const PVRow r) { ref.push_back(r); });
	for (PVRow r : ref) {
		std::cout << r << " ";
	}
	std::cout << std::endl;

	ref.clear();
	a.select_all();
	a.visit_selected_lines([&](const PVRow r) { ref.push_back(r); }, 1);
	for (PVRow r : ref) {
		std::cout << r << " ";
	}
	std::cout << std::endl;

	ref.clear();
	a.visit_selected_lines([&](const PVRow r) { ref.push_back(r); }, 2);
	for (PVRow r : ref) {
		std::cout << r << " ";
	}
	std::cout << std::endl;
	ref.clear();
	a.visit_selected_lines([&](const PVRow r) { ref.push_back(r); }, 69);
	for (PVRow r : ref) {
		std::cout << r << " ";
	}
	std::cout << std::endl;
	ref.clear();
	a.visit_selected_lines([&](const PVRow r) { ref.push_back(r); }, 115);
	for (PVRow r : ref) {
		std::cout << r << " ";
	}
	std::cout << std::endl;
	ref.clear();
	a.visit_selected_lines([&](const PVRow r) { ref.push_back(r); }, 2, 1);
	for (PVRow r : ref) {
		std::cout << r << " ";
	}
	std::cout << std::endl;
	ref.clear();
	a.visit_selected_lines([&](const PVRow r) { ref.push_back(r); }, 4600, 4568);
	for (PVRow r : ref) {
		std::cout << r << " ";
	}
	std::cout << std::endl;
	ref.clear();
	a.visit_selected_lines([&](const PVRow r) { ref.push_back(r); }, 1000000);

	a.select_none();
	b.select_none();
	a.set_line(34, true);
	b.set_line(32 * 5 + 2, true);

	std::cout << "Line 34 set, last nonzero chunk (1) = " << a.get_last_nonzero_chunk_index()
	          << std::endl;
	std::cout << "Line 32*5+2 set, last nonzero chunk (5) = " << b.get_last_nonzero_chunk_index()
	          << std::endl;
	std::cout << "Line 32*5+2 set, last nonzero chunk from 4 (5) = "
	          << b.get_last_nonzero_chunk_index(4) << std::endl;
	std::cout << "Line 32*5+2 set, last nonzero chunk from 5 (5) = "
	          << b.get_last_nonzero_chunk_index(5) << std::endl;
	std::cout << "Line 32*5+2 set, last nonzero chunk from 6 (5) = "
	          << b.get_last_nonzero_chunk_index(6) << std::endl;
	std::cout << "Line 32*5+2 set, last nonzero chunk from 7 (5) = "
	          << b.get_last_nonzero_chunk_index(7) << std::endl;

	std::cout << "Line 32*5+2 set, last nonzero chunk to 4 (5) = "
	          << b.get_last_nonzero_chunk_index(0, 4) << std::endl;
	std::cout << "Line 32*5+2 set, last nonzero chunk to 5 (5) = "
	          << b.get_last_nonzero_chunk_index(0, 5) << std::endl;
	std::cout << "Line 32*5+2 set, last nonzero chunk to 6 (5) = "
	          << b.get_last_nonzero_chunk_index(0, 6) << std::endl;
	std::cout << "Line 32*5+2 set, last nonzero chunk to 7 (5) = "
	          << b.get_last_nonzero_chunk_index(0, 7) << std::endl;

	/**********************************************************************
	***********************************************************************
	*
	* We test all functions inendi_selection_A2A_****
	*
	***********************************************************************
	**********************************************************************/

	/**********************************************************************
	 *
	 * We test inendi_selection_A2A_inverse()
	 *
	 **********************************************************************/

	std::cout << "\nWe test the operator~\n";
	b.select_all();
	BENCH_START(opnot);
	b.select_inverse();
	BENCH_END_TRANSFORM(opnot, "operator ~", sizeof(uint32_t), b.chunk_count());
	for (size_t i = 0; i < SELECTION_COUNT; i++) {
		if (b.get_line(i)) {
			std::cout << " i = " << i << "\n";
			std::cout << "selection test : [" << __LINE__ << "] : operator~ : failed\n";
			return 1;
		}
	}

	/**********************************************************************
	 *
	 * We delete remaining objects
	 *
	 **********************************************************************/
	delete selection;
	delete selection2;
	delete selection3;

	return 0;
}
