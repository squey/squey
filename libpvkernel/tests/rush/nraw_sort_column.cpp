/**
 * \file nraw_sort_column.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <pvkernel/rush/PVNrawDiskBackend.h>
#include <pvkernel/core/picviz_bench.h>
#include <pvkernel/core/PVUnicodeString.h>

#include <string>

#define MIN_SIZE 1
#define MAX_SIZE 256
#define N (1*(1<<14))

PVRush::PVNrawDiskBackend backend;
std::vector<uint32_t> vec;

bool my_predicate(uint32_t i, uint32_t j)
{
	size_t sizei;
	size_t sizej;
	const char* i1 = backend.at(i, 0, sizei);
	const char* j1 = backend.at(j, 0, sizej);
	PVCore::PVUnicodeString s1(i1, sizei);
	PVCore::PVUnicodeString s2(j1, sizej);

	return s1 < s2;
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " path_nraw" << std::endl;
		return 1;
	}

	const char* nraw_path = argv[1];

	backend.init(nraw_path, 1);
	PVLOG_INFO("Writing NRAW...\n");
	char buf[MAX_SIZE];

	vec.reserve(N);
	for (size_t i = 0; i < N; i++) {
		const size_t rand_val = (rand()%(MAX_SIZE-MIN_SIZE+1))+MIN_SIZE;
		snprintf(buf, sizeof(buf), "%lu", rand_val);
		backend.add(0, buf, strlen(buf));
		vec.push_back(i);
	}
	backend.flush();

	PVLOG_INFO("Serial sort... %d values\n", N);

	BENCH_START(std_sort);
	std::sort(vec.begin(), vec.end(), my_predicate);
	BENCH_END(std_sort, "std_sort", 1, 1, sizeof(uint32_t), N);
}
