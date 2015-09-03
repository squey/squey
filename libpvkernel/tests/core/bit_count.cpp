/**
 * \file bit_count.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVBitCount.h>
#include <pvkernel/core/picviz_bench.h>
#include <iostream>

#include <pvkernel/core/picviz_assert.h>

#include <cstdlib>
#include <ctime>

size_t bit_count_ref(uint64_t v)
{
	size_t ret = 0;
	for (int i = 0; i < 64; i++) {
		if ((v & (1ULL<<(i)))) {
			ret++;
		}
	}
	return ret;
}

size_t bit_count_ref(uint32_t v)
{
	size_t ret = 0;
	for (int i = 0; i < 32; i++) {
		if ((v & (1UL<<(i)))) {
			ret++;
		}
	}
	return ret;
}

int main(int argc, char** argv)
{
	// For tests, 2 modes: full-test (takes really a HUGE time, but that's the only
	// way to be sure of our algorithms), or random mode.
	if (argc >= 2) {
		srand(time(NULL));
		size_t n = atoll(argv[1]);

		std::cout << "32-bit tests. It can take a while..." << std::endl;
		for (uint32_t i = 0; i < n / 2; i++) {
			size_t ref = bit_count_ref(i);
			size_t test = PVCore::PVBitCount::bit_count(i);
			PV_VALID(ref, test);
		}
		std::cout << "done" << std::endl;

		std::cout << "64-bit tests (random). It can take a while..." << std::endl;
		for (size_t i = 0; i < n / 2; i++) {
			uint64_t vrand = rand()*rand();
			size_t ref = bit_count_ref(vrand);
			size_t test = PVCore::PVBitCount::bit_count(vrand);

			PV_VALID(ref, test);
		}
		std::cout << "done..." << std::endl;
		return 0;
	} else {
		int ret = 0;

		std::cout << "32-bit tests. It can take a while..." << std::endl;
#pragma omp parallel for
		for (uint32_t i = 0; i < 0xFFFFFFFFUL; i++) {
			size_t ref = bit_count_ref(i);
			size_t test = PVCore::PVBitCount::bit_count(i);
			if (ref != test) {
#pragma omp critical
				std::cerr << "failed at " << i << ": " << test
						  << " but " << ref << " was expected" << std::endl;
				ret = 1;
			}
		}
		PV_ASSERT_VALID(ret == 0);
		std::cout << "done" << std::endl;

		std::cout << "64-bit tests (full). It will take a while..." << std::endl;
#pragma omp parallel for
		for (uint64_t i = 0xFFFFFFFFULL; i < 0xFFFFFFFFFFFFFFFFULL; i++) {
			size_t ref = bit_count_ref(i);
			size_t test = PVCore::PVBitCount::bit_count(i);
			if (ref != test) {
#pragma omp critical
				std::cerr << "failed at " << i << ": " << test
				          << " but " << ref << " was expected" << std::endl;
				ret = 1;
			}
		}
		std::cout << "done..." << std::endl;
		PV_ASSERT_VALID(ret == 0);
		return ret;
	}

}
