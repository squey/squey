/**
 * \file nraw_integrity_test.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <vector>
#include <string>
#include <iostream>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>

#include <pvkernel/rush/PVNrawDiskBackend.h>
#include <pvkernel/core/picviz_assert.h>
#include <pvkernel/core/picviz_bench.h>

constexpr uint32_t COLUMN_COUNT = 2;
constexpr uint32_t ROW_COUNT = 300000;
constexpr uint32_t MAX_FIELD_LENGTH = 512;
static const std::string CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890";

void init_normal_dist()
{
	// Generator engine
	boost::mt19937 rand_gen(boost::mt19937(time(0)));

	// Normal distribution
	typedef boost::random::normal_distribution<uint32_t> normal_dist_t;
	double mean = 0.1;
	double variance = 0.1;
	normal_dist_t normal_dist(mean, variance);
}

struct UniqueNumberGenerator {
	uint32_t operator()() {return _current_number++;}
	uint32_t _current_number = 0;
} unique_number_generator;

std::string random_string(uint32_t length)
{
	std::string result;
	result.resize(length);

	for (uint32_t i = 0; i < length; i++) {
		result[i] = CHARSET[rand() % CHARSET.length()];
	}

	return result;
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " path_nraw" << std::endl;
		return 1;
	}

	srand(time(NULL));

	// Generate nraw
	const char* nraw_path = argv[1];
	PVRush::PVNrawDiskBackend backend;
	backend.init(nraw_path, COLUMN_COUNT);
	std::vector<std::vector<std::string>> random_strings_vect;
	random_strings_vect.resize(COLUMN_COUNT);
	for (uint32_t c = 0 ; c < COLUMN_COUNT; c++) {
		random_strings_vect[c].reserve(ROW_COUNT);
		for (uint32_t r = 0 ; r < ROW_COUNT; r++) {
			const size_t rand_length = 1 + (rand() % (MAX_FIELD_LENGTH-1));
			std::string rs = random_string(rand_length);
			backend.add(c, rs.c_str(), rs.length());
			random_strings_vect[c].push_back(rs);
		}
	}
	backend.flush();

	// Generate shuffled indexes
	std::vector<uint32_t> shuffled_indexes;
	shuffled_indexes.resize(ROW_COUNT);
	std::generate(shuffled_indexes.begin(), shuffled_indexes.end(), unique_number_generator);
	std::random_shuffle(shuffled_indexes.begin(), shuffled_indexes.end());

	size_t size1;
	PVCore::PVSelBitField sel;

	PVLOG_INFO("Verify nraw integrity using 'visit_column'... ");
	BENCH_START(visit_column);
	for (uint32_t c = 0 ; c < COLUMN_COUNT; c++) {
		backend.visit_column2(c, [=](size_t index, const char* buf1, size_t size1)
		{
			PV_ASSERT_VALID(random_strings_vect[c][index].compare(0, size1, buf1) == 0);
		});
	}
	std::cout << "test_passed" << std::endl;
	BENCH_END(visit_column, "visit_column", 1, 1, 1, 1);

	PVLOG_INFO("Verify nraw integrity using 'visit_column2'... ");
	BENCH_START(visit_column2);
	for (uint32_t c = 0 ; c < COLUMN_COUNT; c++) {
		backend.visit_column2(c, [=](size_t index, const char* buf1, size_t size1)
		{
			PV_ASSERT_VALID(random_strings_vect[c][index].compare(0, size1, buf1) == 0);
		});
	}
	std::cout << "test_passed" << std::endl;
	BENCH_END(visit_column2, "visit_column2", 1, 1, 1, 1);

	PVLOG_INFO("Verify nraw integrity using all selected lines 'visit_column2_sel'... ");
	BENCH_START(visit_column2_sel_all);
	sel.select_all();
	for (uint32_t c = 0 ; c < COLUMN_COUNT; c++) {
		backend.visit_column2_sel(c, [&](size_t index, const char* buf1, size_t size1)
		{
			PV_ASSERT_VALID(random_strings_vect[c][index].compare(0, size1, buf1) == 0);
		},
		sel);
	}
	std::cout << "test_passed" << std::endl;
	BENCH_END(visit_column2_sel_all, "visit_column2_sel_all", 1, 1, 1, 1);

	PVLOG_INFO("Verify nraw integrity using random selected lines 'visit_column2_sel'... ");
	BENCH_START(visit_column2_sel_random);
	sel.select_random();
	std::vector<bool> vec_sel;
	vec_sel.resize(ROW_COUNT);
	for(int i = 0; i < ROW_COUNT; i++) {
		vec_sel[i] = sel.get_line_fast(i);
	}
	for (uint32_t c = 0 ; c < COLUMN_COUNT; c++) {
		backend.visit_column2_sel(c, [&](size_t index, const char* buf1, size_t size1)
		{
			PV_ASSERT_VALID(vec_sel[index] == true);
			PV_ASSERT_VALID(random_strings_vect[c][index].compare(0, size1, buf1) == 0);
		},
		sel);
	}
	std::cout << "test_passed" << std::endl;
	BENCH_END(visit_column2_sel_random, "visit_column2_sel_random", 1, 1, 1, 1);

	PVLOG_INFO("Verify nraw integrity using 'visit_column2_tbb'... ");
	BENCH_START(visit_column2_tbb);
	for (uint32_t c = 0 ; c < COLUMN_COUNT; c++) {
		backend.visit_column_tbb(c, [=](size_t index, const char* buf1, size_t size1)
		{
			PV_ASSERT_VALID(random_strings_vect[c][index].compare(0, size1, buf1) == 0);
		});
	}
	std::cout << "test_passed" << std::endl;
	BENCH_END(visit_column2_tbb, "visit_column2_tbb", 1, 1, 1, 1);

	PVLOG_INFO("Verify nraw integrity using all selected lines 'visit_column_tbb_sel'... ");
	BENCH_START(visit_column_tbb_sel_all);
	sel.select_all();
	for (uint32_t c = 0 ; c < COLUMN_COUNT; c++) {
		backend.visit_column_tbb_sel(c, [&](size_t index, const char* buf1, size_t size1)
		{
			PV_ASSERT_VALID(random_strings_vect[c][index].compare(0, size1, buf1) == 0);
		},
		sel);
	}
	std::cout << "test_passed" << std::endl;
	BENCH_END(visit_column_tbb_sel_all, "visit_column_tbb_sel_all", 1, 1, 1, 1);


	PVLOG_INFO("Verify nraw integrity using sequential 'at'... ");
	BENCH_START(sequential_at);
	for (uint32_t index = 0; index < ROW_COUNT; index++) {
		for (uint32_t c = 1 ; c < COLUMN_COUNT; c++) {
			const char* buf1 = backend.at(index, c, size1);
			PV_ASSERT_VALID(random_strings_vect[c][index].compare(0, size1, buf1) == 0);
		}
	}
	std::cout << "test_passed" << std::endl;
	BENCH_END(sequential_at, "sequential_at", 1, 1, 1, 1);

	PVLOG_INFO("Verify nraw integrity using shuffled 'at'... ");
	BENCH_START(shuffled_at);
	for (uint32_t index : shuffled_indexes) {
		for (uint32_t c = 0 ; c < COLUMN_COUNT; c++) {
			const char* buf1 = backend.at(index, c, size1);
			PV_ASSERT_VALID(random_strings_vect[c][index].compare(0, size1, buf1) == 0);
		}
	}
	std::cout << "test_passed" << std::endl;
	BENCH_END(shuffled_at, "shuffled_at", 1, 1, 1, 1);

	PVLOG_INFO("Verify nraw integrity using sequential 'at_no_cache'... ");
	BENCH_START(sequential_at_no_cache);
	for (uint32_t index = 0; index < ROW_COUNT; index++) {
		for (uint32_t c = 0 ; c < COLUMN_COUNT; c++) {
			const char* buf1 = backend.at_no_cache(index, c, size1);
			PV_ASSERT_VALID(random_strings_vect[c][index].compare(0, size1, buf1) == 0);
		}
	}
	std::cout << "test_passed" << std::endl;
	BENCH_END(sequential_at_no_cache, "sequential_at_no_cache", 1, 1, 1, 1);

	PVLOG_INFO("Verify nraw integrity using shuffled 'at_no_cache'... ");
	BENCH_START(shuffled_at_no_cache);
	for (uint32_t index : shuffled_indexes) {
		for (uint32_t c = 0 ; c < COLUMN_COUNT; c++) {
			const char* buf1 = backend.at_no_cache(index, c, size1);
			PV_ASSERT_VALID(random_strings_vect[c][index].compare(0, size1, buf1) == 0);
		}
	}
	std::cout << "test_passed" << std::endl;
	BENCH_END(shuffled_at_no_cache, "shuffled_at_no_cache", 1, 1, 1, 1);


	PVLOG_INFO("Verify nraw integrity using shuffled parallel 'at_no_cache'... ");
	BENCH_START(shuffled_parallel_at_no_cache);
	for (uint32_t c = 0 ; c < COLUMN_COUNT; c++) {
#pragma omp parallel for
		for (uint32_t r = 0 ; r < ROW_COUNT; r++) {
			uint32_t index = shuffled_indexes[r];
			size_t size;
			const char* buf1 = backend.at_no_cache(index, c, size);
			PV_ASSERT_VALID(random_strings_vect[c][index].compare(0, size, buf1) == 0);
		}
	}
	std::cout << "test_passed" << std::endl;
	BENCH_END(shuffled_parallel_at_no_cache, "shuffled_parallel_at_no_cache", 1, 1, 1, 1);

	return 0;
}




