/**
 * \file nraw_sort_column.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <pvkernel/core/picviz_bench.h>
#include <pvkernel/core/picviz_assert.h>
#include <pvkernel/rush/PVNrawDiskBackend.h>
#include <pvkernel/core/PVUnicodeString.h>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_sort.h>
#include <tbb/scalable_allocator.h>

#include <array>
#include <iostream>

static const char* g_charset = "azertyuiopqsdfghjklmwxcvbnAZERTYUIOPQSDFGHJKLMWXCVBN0123456789";

#define MIN_RAND 100
#define MAX_RAND 500

class NrawColumnStableSort
{
public:
	NrawColumnStableSort(PVRush::PVNrawDiskBackend const& backend, PVCol col):
		_backend(backend), _col(col)
	{ }

public:
	inline bool operator()(const PVRow i1, const PVRow i2) const
	{
		const PVCol col = _col;
		PVRush::PVNrawDiskBackend const& backend = _backend;

		size_t size_1, size_2;
		char* str1 = &_tmp_buf.local()[0];
		backend.at_no_cache(i1, col, size_1, str1);
		const char* str2 = backend.at_no_cache(i2, col, size_2);

		PVCore::PVUnicodeString uni1(str1, size_1);
		PVCore::PVUnicodeString uni2(str2, size_2);

		int ret = uni1.compare(uni2);
		if (ret == 0) {
			return i1 < i2;
		}

		return ret;
	}

private:
	PVRush::PVNrawDiskBackend const& _backend;
	mutable tbb::enumerable_thread_specific<std::array<char, PVRush::PVNrawDiskBackend::READ_BUFFER_SIZE+PVRush::PVNrawDiskBackend::BUF_ALIGN>> _tmp_buf;
	PVCol _col;
};

std::string random_string(uint32_t length)
{
	std::string result;
	result.resize(length);

	for (uint32_t i = 0; i < length; i++) {
		result[i] = g_charset[rand()%(sizeof(g_charset)/sizeof(char))];
	}

	return std::move(result);
}

template <typename Iterator>
void show_nraw(PVRush::PVNrawDiskBackend const& backend, PVCol const col, Iterator const& begin, Iterator const& end)
{
	for (Iterator it = begin; it != end; it++) {
		size_t sbuf;
		const char* buf = backend.at_no_cache(*it, col, sbuf);
		std::cout.write(buf, sbuf);
		std::cout << std::endl;
	}
}

int main(int argc, char** argv)
{
	if (argc < 3) {
		std::cerr << "Usage: " << argv[0] << " path_nraw nrows" << std::endl;
		return 1;
	}

	const char* nraw_path = argv[1];
	const PVRow nrows = atoll(argv[2]);

	PVRush::PVNrawDiskBackend backend;
	backend.init(nraw_path, 1);
	PVLOG_INFO("Writing NRAW...\n");

	std::vector<std::string> random_strings_vect;
	random_strings_vect.reserve(nrows);
	for (size_t r = 0 ; r < nrows; r++) {
		const size_t rand_length = MIN_RAND + (rand() % (MAX_RAND-MIN_RAND+1));
		std::string rs = random_string(rand_length);
		backend.add(0, rs.c_str(), rs.length());
		random_strings_vect.push_back(rs);
	}

	// TBB SSD parallel sort
	PVRow* indexes;
	posix_memalign((void**) &indexes, 16, sizeof(PVRow)*nrows);
	for (PVRow i = 0; i < nrows; i++) {
		indexes[i] = i;
	}

	show_nraw(backend, 0, indexes, indexes + nrows);

	NrawColumnStableSort sort_obj(backend, 0);

	BENCH_START(tbbsort);
	tbb::parallel_sort(indexes, indexes + nrows, sort_obj);
	BENCH_END(tbbsort, "stable-sort-tbb", 1, 1, 1, 1);

	show_nraw(backend, 0, indexes, indexes + nrows);

	for (PVRow i = 0; i < nrows; i++) {
		indexes[i] = i;
	}
	BENCH_START(tbbsort2);
	tbb::parallel_sort(indexes, indexes + nrows, sort_obj);
	BENCH_END(tbbsort2, "stable-sort-tbb-2", 1, 1, 1, 1);

	free(indexes);

	return 0;
}
