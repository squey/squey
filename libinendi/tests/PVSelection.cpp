/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <stdlib.h>
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

	PVRow last_index;
	PVRow count;

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

	std::cout << "we test get_number_of_selected_lines_in_range() and select_all()\n";
	last_index = SELECTION_COUNT;

	selection = new Inendi::PVSelection(SELECTION_COUNT);
	selection->select_all();
	count = selection->get_number_of_selected_lines_in_range(0, 1);
	std::cout << "count should be 1: " << count << std::endl;
	if (count != 1) {
		return 1;
	}
	count = selection->get_number_of_selected_lines_in_range(0, 2);
	std::cout << "count should be 2: " << count << std::endl;
	if (count != 2) {
		return 1;
	}
	count = selection->get_number_of_selected_lines_in_range(10, 15);
	std::cout << "count should be 5: " << count << std::endl;
	if (count != 5) {
		return 1;
	}
	count = selection->get_number_of_selected_lines_in_range(0, 64);
	std::cout << "count should be 64: " << count << std::endl;
	if (count != 64) {
		return 1;
	}
	count = selection->get_number_of_selected_lines_in_range(0, 65);
	std::cout << "count should be 65: " << count << std::endl;
	if (count != 65) {
		return 1;
	}
	count = selection->get_number_of_selected_lines_in_range(5, 65);
	std::cout << "count should be 60: " << count << std::endl;
	if (count != 60) {
		return 1;
	}
	count = selection->get_number_of_selected_lines_in_range(0, last_index);
	std::cout << "is " << SELECTION_COUNT << " = to " << count << " ?\n\n";
	if (count != SELECTION_COUNT) {
		return 1;
	}

	std::cout << "we test select_even()\n";
	selection->select_even();
	count = selection->get_number_of_selected_lines_in_range(0, last_index);
	std::cout << "is " << SELECTION_COUNT / 2 << " = to " << count << " ?\n\n";
	if (count != SELECTION_COUNT / 2) {
		return 1;
	}

	std::cout << "we test select_odd()\n";
	selection->select_odd();
	count = selection->get_number_of_selected_lines_in_range(0, last_index);
	std::cout << "is " << SELECTION_COUNT / 2 << " = to " << count << " ?\n\n";
	if (count != SELECTION_COUNT / 2) {
		return 1;
	}

	std::cout << "we test get_number_of_selected_lines_in_range(a,b)\n";
	count = selection->get_number_of_selected_lines_in_range(0, 100);
	std::cout << "is 50 = to " << count << " ?\n\n";
	if (count != 50) {
		return 1;
	}

	// Test of is_empty
	for (int i = 0; i < 256; i++) {
		a.select_none();
		a.set_bit_fast(i);
		PV_ASSERT_VALID(not a.is_empty());
	}

	// Test of is_empty_between ([a,b[)
	a.select_none();
	a.set_bit_fast(6);
	PV_ASSERT_VALID(a.is_empty_between(0, 6));
	PV_ASSERT_VALID(a.is_empty_between(7, SELECTION_COUNT));
	a.set_bit_fast(65);
	PV_ASSERT_VALID(a.is_empty_between(66, SELECTION_COUNT));
	a.set_bit_fast(88);
	PV_ASSERT_VALID(a.is_empty_between(66, 88));
	PV_ASSERT_VALID(a.is_empty_between(66, 89) == false);
	PV_ASSERT_VALID(a.is_empty_between(65, 88) == false);
	PV_ASSERT_VALID(a.is_empty_between(89, 10000));
	PV_ASSERT_VALID(a.is_empty_between(88, 10000) == false);
	a.set_bit_fast(1024);
	PV_ASSERT_VALID(a.is_empty_between(100, 10000) == false);
	a.set_bit_fast(5);
	a.set_bit_fast(63);
	a.set_bit_fast(64);
	a.visit_selected_lines([&](const PVRow r) { std::cout << r << " "; });
	std::cout << std::endl;

	// Test of C++0x features
	a.select_even();
	b.select_odd();

	Inendi::PVSelection c = std::move(a & b);
	PVLOG_INFO("a: %p , b = %p , c = %p\n", &a, &b, &c);
	std::cout << "PVSelection should be empty: PVSelection::is_empty() = " << c.is_empty()
	          << std::endl;

	c = a;
	c.and_optimized(b);
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
	c.or_optimized(b);
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
	c.or_optimized(b);
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
	a = ~b;

	size_t i, good;
	for (i = 0; i < SELECTION_COUNT; i++) {
		good = (a.get_line(i));
		if (not good) {
			std::cout << " i = " << i << ", and state of line n is : " << good << "\n";
			std::cout << "selection test : [" << __LINE__ << "] : operator~ : failed\n";
			return 1;
		}
	}

	/**********************************************************************
	*
	* We delete remaining objects
	*
	**********************************************************************/
	// delete(selection);
	delete (selection2);
	delete (selection3);

	return 0;

	std::cout << "#### MAPPED ####\n";
}
