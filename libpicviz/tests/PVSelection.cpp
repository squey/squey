/**
 * \file PVSelection.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <stdlib.h>
#include <pvkernel/core/picviz_assert.h>
#include <pvkernel/core/picviz_bench.h>

#include <picviz/PVSelection.h>
#include <iostream>


int main(void)
{

// #include "test-env.h"

	Picviz::PVSelection *selection;
	Picviz::PVSelection *selection2 = new Picviz::PVSelection();
	Picviz::PVSelection *selection3 = new Picviz::PVSelection();
	Picviz::PVSelection a;
	Picviz::PVSelection b;
	
	

	int i, good;
	int line_index;

	PVRow last_index;
	PVRow count;

	/**********************************************************************
	*
	* We test creation and deletion of a picviz_selection_t
	*
	**********************************************************************/

	selection = new Picviz::PVSelection();
	delete(selection);


	a.select_none();
	BENCH_START(sel);
	std::cout << "get_last_nonzero_chunk_index on empty sel: " << a.get_last_nonzero_chunk_index() << std::endl;
	BENCH_END(sel, "bench", sizeof(uint32_t), PICVIZ_SELECTION_NUMBER_OF_CHUNKS, 1, 1);

	a.select_all();
	BENCH_START(self);
	std::cout << "get_last_nonzero_chunk_index on full sel (should be " << (PICVIZ_SELECTION_NUMBER_OF_CHUNKS-1) << "): " << a.get_last_nonzero_chunk_index() << std::endl;
	BENCH_END(self, "bench", sizeof(uint32_t), PICVIZ_SELECTION_NUMBER_OF_CHUNKS, 1, 1);

	a.select_none();
	a.set_bit_fast(4);
	std::cout << "get_last_nonzero_chunk_index with bit 4: " << a.get_last_nonzero_chunk_index() << std::endl;

	a.set_bit_fast(44);
	std::cout << "get_last_nonzero_chunk_index with bit 44: " << a.get_last_nonzero_chunk_index() << std::endl;

	a.set_bit_fast(32*4);
	std::cout << "get_last_nonzero_chunk_index with bit 32*4: " << a.get_last_nonzero_chunk_index() << std::endl;

	a.visit_selected_lines([&](PVRow r) { std::cout << r << "," ; } );
	std::cout << std::endl;
	//a.visit_selected_lines_sse([&](PVRow r) { std::cout << r << "," ; } );
	//std::cout << std::endl;

	a.select_none();
#define NLINES_TEST 10000
	for (int i = 0; i < NLINES_TEST; i++) {
		a.set_bit_fast(rand()%(PICVIZ_LINES_MAX/4));
	}
	std::vector<PVRow> ref,test;
	ref.reserve(NLINES_TEST); test.reserve(NLINES_TEST);
	BENCH_START(ref);
	a.visit_selected_lines([&](PVRow r) { ref.push_back(r); });
	BENCH_END(ref, "visit ref", sizeof(uint32_t), PICVIZ_SELECTION_NUMBER_OF_CHUNKS, sizeof(PVRow), ref.size());
	//BENCH_START(sse);
	//a.visit_selected_lines_sse([&](PVRow r) { test.push_back(r); });
	//BENCH_END(sse, "visit sse", sizeof(uint32_t), PICVIZ_SELECTION_NUMBER_OF_CHUNKS, sizeof(PVRow), test.size());

	std::cout << "Visit sse test: " << (ref == test) << std::endl;

	/**********************************************************************
	*
	* We test all Generic functions
	*
	**********************************************************************/

	std::cout << "we test get_number_of_selected_lines_in_range() and select_all()\n";
	last_index = PICVIZ_LINES_MAX;

	selection = new Picviz::PVSelection();
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
	std::cout << "is " << PICVIZ_LINES_MAX << " = to " << count << " ?\n\n";
	if (count != PICVIZ_LINES_MAX) {
		return 1;
	}

	std::cout << "we test select_even()\n";
	selection->select_even();
	count = selection->get_number_of_selected_lines_in_range(0, last_index);
	std::cout << "is " << PICVIZ_LINES_MAX/2 << " = to " << count << " ?\n\n";
	if (count != PICVIZ_LINES_MAX/2) {
		return 1;
	}
	
	std::cout << "we test select_odd()\n";
	selection->select_odd();
	count = selection->get_number_of_selected_lines_in_range(0, last_index);
	std::cout << "is " << PICVIZ_LINES_MAX/2 << " = to " << count << " ?\n\n";
	if (count != PICVIZ_LINES_MAX/2) {
		return 1;
	}

	std::cout << "we test get_number_of_selected_lines_in_range(a,b)\n";
	count = selection->get_number_of_selected_lines_in_range(0, 100);
	std::cout << "is 50 = to " << count << " ?\n\n";
	if (count != 50) {
		return 1;
	}

	// Test of is_empty_between ([a,b[)
	a.select_none();
	a.set_bit_fast(6);
	ASSERT_VALID(a.is_empty_between(0, 6));
	ASSERT_VALID(a.is_empty_between(7, PICVIZ_LINES_MAX));
	a.set_bit_fast(65);
	ASSERT_VALID(a.is_empty_between(66, PICVIZ_LINES_MAX));
	a.set_bit_fast(88);
	ASSERT_VALID(a.is_empty_between(66, 88));
	ASSERT_VALID(a.is_empty_between(66, 89) == false);
	ASSERT_VALID(a.is_empty_between(65, 88) == false);
	ASSERT_VALID(a.is_empty_between(89, 10000));
	ASSERT_VALID(a.is_empty_between(88, 10000) == false);
	a.set_bit_fast(1024);
	ASSERT_VALID(a.is_empty_between(100, 10000) == false);
	a.set_bit_fast(5);
	a.set_bit_fast(63);
	a.set_bit_fast(64);
	a.visit_selected_lines([&](const PVRow r) { std::cout << r << " "; });
	std::cout << std::endl;
	

	// Test of C++0x features
	a.select_even();
	b.select_odd();

	Picviz::PVSelection c = std::move(a & b);
	PVLOG_INFO("a: %p , b = %p , c = %p\n", &a, &b, &c);
	std::cout << "PVSelection should be empty: PVSelection::is_empty() = " << c.is_empty() << std::endl;

	c = a;
	c.and_optimized(b);
	std::cout << "PVSelection should be empty: PVSelection::is_empty() = " << c.is_empty() << std::endl;

	c = a;
	BENCH_START(original_or);
	c |= b;
	BENCH_END_TRANSFORM(original_or, "Original OR", sizeof(uint32_t), PICVIZ_SELECTION_NUMBER_OF_CHUNKS);
	std::cout << "Last chunk should be " << PICVIZ_SELECTION_NUMBER_OF_CHUNKS-1 << " : " << c.get_last_nonzero_chunk_index() << std::endl;

	c = a;
	BENCH_START(opt_or);
	c.or_optimized(b);
	BENCH_END_TRANSFORM(opt_or, "Opt OR", sizeof(uint32_t), PICVIZ_SELECTION_NUMBER_OF_CHUNKS);
	std::cout << "Last chunk should be " << PICVIZ_SELECTION_NUMBER_OF_CHUNKS-1 << " : " << c.get_last_nonzero_chunk_index() << std::endl;

	b.select_none();

	c = a;
	BENCH_START(original_or2);
	c |= b;
	BENCH_END_TRANSFORM(original_or2, "Original OR", sizeof(uint32_t), PICVIZ_SELECTION_NUMBER_OF_CHUNKS);
	std::cout << "Last chunk : " << c.get_last_nonzero_chunk_index() << std::endl;

	c = a;
	BENCH_START(opt_or2);
	c.or_optimized(b);
	BENCH_END_TRANSFORM(opt_or2, "Opt OR", sizeof(uint32_t), PICVIZ_SELECTION_NUMBER_OF_CHUNKS);
	std::cout << "Last chunk : " << c.get_last_nonzero_chunk_index() << std::endl;

	a.select_none();
	a.set_bit_fast(1);
	ref.clear();
	a.visit_selected_lines([&](const PVRow r) { ref.push_back(r); });
	for (PVRow r: ref) { std::cout << r << " "; }
	std::cout << std::endl;

	ref.clear();
	a.select_all();
	a.visit_selected_lines([&](const PVRow r) { ref.push_back(r); }, 1);
	for (PVRow r: ref) { std::cout << r << " "; }
	std::cout << std::endl;

	
	ref.clear();
	a.visit_selected_lines([&](const PVRow r) { ref.push_back(r); }, 2);
	for (PVRow r: ref) { std::cout << r << " "; }
	std::cout << std::endl;
	ref.clear();
	a.visit_selected_lines([&](const PVRow r) { ref.push_back(r); }, 69);
	for (PVRow r: ref) { std::cout << r << " "; }
	std::cout << std::endl;
	ref.clear();
	a.visit_selected_lines([&](const PVRow r) { ref.push_back(r); }, 115);
	for (PVRow r: ref) { std::cout << r << " "; }
	std::cout << std::endl;
	ref.clear();
	a.visit_selected_lines([&](const PVRow r) { ref.push_back(r); }, 2, 1);
	for (PVRow r: ref) { std::cout << r << " "; }
	std::cout << std::endl;
	ref.clear();
	a.visit_selected_lines([&](const PVRow r) { ref.push_back(r); }, 4600, 4568);
	for (PVRow r: ref) { std::cout << r << " "; }
	std::cout << std::endl;
	ref.clear();
	a.visit_selected_lines([&](const PVRow r) { ref.push_back(r); }, 1000000);
	

	a.select_none();
	b.select_none();
	a.set_line(34, true);
	b.set_line(32*5+2, true);

	std::cout << "Line 34 set, last nonzero chunk (1) = " << a.get_last_nonzero_chunk_index() << std::endl;
	std::cout << "Line 32*5+2 set, last nonzero chunk (5) = " << b.get_last_nonzero_chunk_index() << std::endl;
	std::cout << "Line 32*5+2 set, last nonzero chunk from 4 (5) = " << b.get_last_nonzero_chunk_index(4) << std::endl;
	std::cout << "Line 32*5+2 set, last nonzero chunk from 5 (5) = " << b.get_last_nonzero_chunk_index(5) << std::endl;
	std::cout << "Line 32*5+2 set, last nonzero chunk from 6 (5) = " << b.get_last_nonzero_chunk_index(6) << std::endl;
	std::cout << "Line 32*5+2 set, last nonzero chunk from 7 (5) = " << b.get_last_nonzero_chunk_index(7) << std::endl;

	std::cout << "Line 32*5+2 set, last nonzero chunk to 4 (5) = " << b.get_last_nonzero_chunk_index(0, 4) << std::endl;
	std::cout << "Line 32*5+2 set, last nonzero chunk to 5 (5) = " << b.get_last_nonzero_chunk_index(0, 5) << std::endl;
	std::cout << "Line 32*5+2 set, last nonzero chunk to 6 (5) = " << b.get_last_nonzero_chunk_index(0, 6) << std::endl;
	std::cout << "Line 32*5+2 set, last nonzero chunk to 7 (5) = " << b.get_last_nonzero_chunk_index(0, 7) << std::endl;

	/*
	std::cout << "Checking only one line set in range [1024, 1088]" << std::endl;
	for (int i = 0; i < (PICVIZ_SELECTION_CHUNK_SIZE * 2) + 1; ++i) {
		PVRow r = 1024 + i;
		a.select_none();
		a.set_line(r, true);
		ssize_t expected = r / PICVIZ_SELECTION_CHUNK_SIZE;
		ssize_t res = a.get_last_nonzero_chunk_index(0, 64);
		if (res != expected) {
			std::cout << "Test fails with line " << r
			          << " set and last nonzero chunk to 64: returns "
			          << res << " but " << expected << " expected" << std::endl;
		}
	}*/

	/**********************************************************************
	***********************************************************************
	*
	* We test all functions picviz_selection_A2A_****
	*
	***********************************************************************
	**********************************************************************/


	/**********************************************************************
	*
	* We test picviz_selection_A2A_inverse()
	*
	**********************************************************************/

	std::cout << "\nWe test the operator~\n";
	b.select_all();
	BENCH_START(opnot);
	b.select_inverse();
	BENCH_END_TRANSFORM(opnot, "operator ~", sizeof(uint32_t), PICVIZ_SELECTION_NUMBER_OF_CHUNKS);
	a = ~b;

	for (i=0; i<PICVIZ_LINES_MAX; i++) {
		good = (a.get_line(i));
		if (good) {
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
	//delete(selection);
	delete(selection2);
	delete(selection3);

#if 0
	Picviz::PVSelection *a = new Picviz::PVSelection();
	Picviz::PVSelection b;
	a->select_all();
	b.select_odd();
	*a = ~b;
	delete a;
#endif

	return 0;


	std::cout << "#### MAPPED ####\n";

}
