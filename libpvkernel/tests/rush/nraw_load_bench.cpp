/**
 * \file nraw_load_bench.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <iostream>
#include <vector>

#include <tbb/tick_count.h>

#include <pvkernel/rush/PVNrawDiskBackend.h>
#include <pvkernel/core/picviz_bench.h>

#define N 2000000000
#define M 1000000

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " path_nraw" << std::endl;
		return 1;
	}

	const char* nraw_path = argv[1];
	PVRush::PVNrawDiskBackend backend;

	backend.set_direct_mode(false);

	backend.init(nraw_path, 1);
	backend.load_index_from_disk();
	backend.print_indexes();

	size_t ret;

	// Sequential string vector
	std::vector<std::string> vec;
	vec.reserve(M);
	for (int i = 0 ; i < M; i++) {
		std::stringstream st;
		st << i << " ";
		vec.push_back(st.str());
	}

	// Shuffled vector
	std::vector<unsigned int> shuffled_fields_sequence;
	std::vector<std::string> shuffled_fields_sequence_string;
	shuffled_fields_sequence.reserve(M);
	shuffled_fields_sequence_string.reserve(M);
	for (unsigned int i = 0; i < M; i++) {
		shuffled_fields_sequence.push_back(rand() % N);
		std::stringstream st;
		st << i;
		shuffled_fields_sequence_string.push_back(st.str());
	}

	// Sequential with cache
	{
	bool test_passed = true;
	BENCH_START(nraw_sequential_with_cache);
	for (int i = 0 ; i < M; i++) {
		const char* field = backend.at(i, 0, ret);
		test_passed &= (strcmp(field, vec[i].c_str()) == 0);
		std::cout << "field=" << field << " vec[i].c_str()=" << vec[i].c_str() << std::endl;
	}
	BENCH_END(nraw_sequential_with_cache, "nraw_sequential_with_cache", 1, 1, 1, 1);
	std::cout << "test passed: " << std::boolalpha << test_passed << std::endl;
	}

	// Sequential visit
	{
	BENCH_START(nraw_sequential_visit);
	bool test_passed = true;
	int i = 0;
	backend.visit_column2(0, [=,&i,&test_passed](size_t r, const char* field, size_t n)
	{
		PV_UNUSED(r);
		PV_UNUSED(n);
		test_passed &= (strcmp(field, vec[i].c_str()) == 0);
		i++;
	});
	BENCH_END(nraw_sequential_visit, "nraw_sequential_visit", 1, 1, 1, 1);
	std::cout << "test passed: " << std::boolalpha << test_passed << std::endl;
	}

	// Sequential visit tbb
	{
	BENCH_START(nraw_sequential_visit_tbb);
	bool test_passed = true;
	int i = 0;
	backend.visit_column_tbb(0, [=,&i,&test_passed](size_t r, const char* field, size_t n)
	{
		PV_UNUSED(r);
		PV_UNUSED(n);
		test_passed &= (strcmp(field, vec[i].c_str()) == 0);
		i++;
	});
	BENCH_END(nraw_sequential_visit_tbb, "nraw_sequential_visit_tbb", 1, 1, 1, 1);
	std::cout << "test passed: " << std::boolalpha << test_passed << std::endl;
	}

	// Sequential without cache
	/*{
	bool test_passed = true;
	BENCH_START(nraw_sequential_without_cache);
	for (int i = 0 ; i < RAND && test_passed; i++) {
		const char* field = backend.at_no_cache(i, 0, ret);
		test_passed &= (strcmp(field, vec[i].c_str()) == 0);
	}
	BENCH_END(nraw_sequential_without_cache, "nraw_sequential_without_cache", 1, 1, 1, 1);
	std::cout << "test passed: " << std::boolalpha << test_passed << std::endl;
	}*/

	// Random with cache
	{
	bool test_passed = true;
	BENCH_START(nraw_random_with_cache);
	for (unsigned int i : shuffled_fields_sequence) {
		const char* field = backend.at(i, 0, ret);
		std::stringstream st;
		st << i;
		test_passed &= (strcmp(field, st.str().c_str()) == 0);
	}
	BENCH_END(nraw_random_with_cache, "nraw_random_with_cache", 1, 1, 1, 1);
	std::cout << "test passed: " << std::boolalpha << test_passed << std::endl;
	}

	// Random without cache
	{
	bool test_passed = true;
	BENCH_START(nraw_random_without_cache);
	for (unsigned int i : shuffled_fields_sequence) {
		const char* field = backend.at_no_cache(i, 0, ret);
		std::stringstream st;
		st << i;
		test_passed &= (strcmp(field, st.str().c_str()) == 0);
	}
	BENCH_END(nraw_random_without_cache, "nraw_random_without_cache", 1, 1, 1, 1);
	std::cout << "test passed: " << std::boolalpha << test_passed << std::endl;
	}

	// Random omp without cache
	{
	bool test_passed = true;
	BENCH_START(nraw_random_omp_without_cache);
#pragma omp parallel for
	for (unsigned int j = 0 ; j < shuffled_fields_sequence.size(); j++) {
		int i = shuffled_fields_sequence[j];
		const char* field = backend.at_no_cache(i, 0, ret);
		std::stringstream st;
		st << i;
		test_passed &= (strcmp(field, st.str().c_str()) == 0);
	}
	BENCH_END(nraw_random_omp_without_cache, "nraw_random_omp_without_cache", 1, 1, 1, 1);
	std::cout << "test passed: " << std::boolalpha << test_passed << std::endl;
	double time = BENCH_END_TIME(nraw_random_omp_without_cache);
	std::cout << "op/time=" << shuffled_fields_sequence.size()/time << std::endl;
	}
}

