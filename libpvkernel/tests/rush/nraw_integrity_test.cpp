/**
 * \file nraw_integrity_test.cpp
 *
 * Copyright (C) Picviz Labs 2010-2013
 */

#include <vector>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <memory>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>

#include <pvkernel/rush/PVNrawDiskBackend.h>
#include <pvkernel/core/picviz_assert.h>
#include <pvkernel/core/picviz_bench.h>
#include <pvkernel/core/PVDirectory.h>

#include <QDir>

constexpr uint32_t COLUMN_COUNT = 2;
constexpr uint32_t ROW_COUNT = 500;
constexpr uint32_t MAX_FIELD_LENGTH = 512;
static const std::string CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890";

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

struct abstract_generator
{
	virtual uint32_t operator()() { return 0; }
	virtual ~abstract_generator() {}
	virtual std::string to_string() { return "default"; }
};

struct fixed_generator : public abstract_generator
{
	fixed_generator(uint32_t val) : value(val)  {}
	virtual uint32_t operator()() { return value; }
	std::string to_string() { return QString("fixed generator [%1]").arg(value).toStdString(); }
	uint32_t value;
};

struct increasing_generator : public abstract_generator
{
	increasing_generator(uint32_t mi, uint32_t ma) : min(mi), max(ma), _current(min) {}
	virtual uint32_t operator()() { return min + (_current++ % (uint32_t)(max - min + 1)); }
	std::string to_string() { return QString("increasing generator in range [%1;%2]").arg(min).arg(max).toStdString(); }
	uint32_t min;
	uint32_t max;
	uint32_t _current;
};

struct random_generator : public abstract_generator
{
	random_generator(uint32_t mi, uint32_t ma) : min(mi), max(ma) {}
	virtual uint32_t operator()() { return min + (rand() % (uint32_t)(max - min + 1)); }
	std::string to_string() { return QString("random generator in range [%1;%2]").arg(min).arg(max).toStdString(); }
	uint32_t min;
	uint32_t max;
};

struct random_gaussian_generator : public abstract_generator
{
	typedef boost::random::normal_distribution<double> normal_dist_t;
	random_gaussian_generator(uint32_t mi, uint32_t ma, double m, double v) : min(mi), max(ma), mean(m), variance(v), _rand_gen(boost::mt19937(time(0))), _normal_dist(mean, variance) {}
	virtual uint32_t operator()() { return  min + ((uint32_t) _normal_dist(_rand_gen)*max) % (max - min + 1); }
	std::string to_string() { return QString("normal dist range [%1;%2](mean=%3, variance=%4)").arg(min).arg(max).arg(mean).arg(variance).toStdString(); }
	uint32_t min;
	uint32_t max;
	double mean;
	double variance;
	boost::mt19937 _rand_gen;
	normal_dist_t _normal_dist;
};

QString g_nraw_folder;

void cleanup()
{
	std::cout << std::endl << "cleaning up " << qPrintable(g_nraw_folder) << " folder" << std::endl;
	PVCore::PVDirectory::remove_rec(g_nraw_folder);
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " path_nraw" << std::endl;
		return 1;
	}
	g_nraw_folder = PVCore::PVDirectory::temp_dir(QDir(argv[1]), "nraw-disk-backend-test-integrity-XXXXXX");

	QDir dir;
	dir.mkdir(g_nraw_folder);
	atexit(cleanup);
	srand(time(NULL));

	typedef std::shared_ptr<abstract_generator> abstract_generator_sp;
	std::vector<abstract_generator_sp> ranges;
	ranges.push_back(abstract_generator_sp(new fixed_generator(0)));
	ranges.push_back(abstract_generator_sp(new fixed_generator(1)));
	ranges.push_back(abstract_generator_sp(new fixed_generator(2)));
	ranges.push_back(abstract_generator_sp(new fixed_generator(3)));
	ranges.push_back(abstract_generator_sp(new fixed_generator(4)));
	ranges.push_back(abstract_generator_sp(new fixed_generator(MAX_FIELD_LENGTH-1)));
	ranges.push_back(abstract_generator_sp(new fixed_generator(MAX_FIELD_LENGTH)));
	ranges.push_back(abstract_generator_sp(new increasing_generator(0, MAX_FIELD_LENGTH)));
	ranges.push_back(abstract_generator_sp(new random_generator(0, 4)));
	ranges.push_back(abstract_generator_sp(new random_generator(MAX_FIELD_LENGTH-4, MAX_FIELD_LENGTH)));
	ranges.push_back(abstract_generator_sp(new random_generator(128, MAX_FIELD_LENGTH)));
	ranges.push_back(abstract_generator_sp(new random_generator(0, MAX_FIELD_LENGTH)));
	ranges.push_back(abstract_generator_sp(new random_gaussian_generator(0, MAX_FIELD_LENGTH, 0.3, 0.1)));

	for (uint32_t t = 0; t < ranges.size(); t++) {

		PVLOG_INFO("\n\nString length in %s:\n", ranges[t]->to_string().c_str());

		// Generate nraw
		PVRush::PVNrawDiskBackend backend;
		backend.init(qPrintable(g_nraw_folder), COLUMN_COUNT);
		std::vector<std::vector<std::string>> random_strings_vect;
		random_strings_vect.resize(COLUMN_COUNT);
		for (uint32_t c = 0 ; c < COLUMN_COUNT; c++) {
			random_strings_vect[c].reserve(ROW_COUNT);
			for (uint32_t r = 0 ; r < ROW_COUNT; r++) {
				const size_t rand_length = (*ranges[t])();
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
		for(uint32_t i = 0; i < ROW_COUNT; i++) {
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
				size_t size = 0;
				const char* buf1 = backend.at_no_cache(index, c, size);
				PV_ASSERT_VALID(random_strings_vect[c][index].compare(0, size, buf1) == 0);
			}
		}
		std::cout << "test_passed" << std::endl;
		BENCH_END(shuffled_parallel_at_no_cache, "shuffled_parallel_at_no_cache", 1, 1, 1, 1);

	}

	return 0;
}




