/**
 * \file nraw_bench.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <iostream>
#include <vector>

#include <tbb/tick_count.h>

#include <pvkernel/rush/PVNrawDiskBackend.h>
#include <pvkernel/core/picviz_bench.h>

#define N (5000000)
#define LATENCY_N N

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


	std::vector<std::string> vec;
	vec.reserve(N);
	for (int i = 0 ; i < N; i++) {
		std::stringstream st;
		st << i << " ";
		vec.push_back(st.str());
		backend.add(0, st.str().c_str(), st.str().length());
	}
	backend.flush();

	size_t ret;

	std::vector<unsigned int> shuffled_fields_sequence;
	shuffled_fields_sequence.reserve(LATENCY_N);
	for (unsigned int i = 0; i < LATENCY_N; i++) {
		shuffled_fields_sequence.push_back(i);
	}
	std::random_shuffle(shuffled_fields_sequence.begin(), shuffled_fields_sequence.end());

	// Sequential with cache
	{
	bool test_passed = true;
	BENCH_START(nraw_sequential_with_cache);
	for (int i = 0 ; i < N && test_passed; i++) {
		const char* field = backend.at(i, 0, ret);
		test_passed &= (strcmp(field, vec[i].c_str()) == 0);
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
		test_passed &= (strcmp(field, vec[i].c_str()) == 0);
		i++;
	});
	BENCH_END(nraw_sequential_visit_tbb, "nraw_sequential_visit_tbb", 1, 1, 1, 1);
	std::cout << "test passed: " << std::boolalpha << test_passed << std::endl;
	}

	// Sequential without cache
	{
	bool test_passed = true;
	BENCH_START(nraw_sequential_without_cache);
	for (int i = 0 ; i < N && test_passed; i++) {
		const char* field = backend.at_no_cache(i, 0).c_str();
		test_passed &= (strcmp(field, vec[i].c_str()) == 0);
	}
	BENCH_END(nraw_sequential_without_cache, "nraw_sequential_without_cache", 1, 1, 1, 1);
	std::cout << "test passed: " << std::boolalpha << test_passed << std::endl;
	}

	// Random with cache
	{
	bool test_passed = true;
	BENCH_START(nraw_random_with_cache);
	for (unsigned int i : shuffled_fields_sequence) {
		const char* field = backend.at(i, 0, ret);
		test_passed &= (strcmp(field, vec[i].c_str()) == 0);
	}
	BENCH_END(nraw_random_with_cache, "nraw_random_with_cache", 1, 1, 1, 1);
	std::cout << "test passed: " << std::boolalpha << test_passed << std::endl;
	}

	// Random without cache
	{
	bool test_passed = true;
	BENCH_START(nraw_random_without_cache);
	for (unsigned int i : shuffled_fields_sequence) {
		const char* field = backend.at_no_cache(i, 0).c_str();
		test_passed &= (strcmp(field, vec[i].c_str()) == 0);
	}
	BENCH_END(nraw_random_without_cache, "nraw_random_without_cache", 1, 1, 1, 1);
	std::cout << "test passed: " << std::boolalpha << test_passed << std::endl;
	}

	backend.clear();
}

