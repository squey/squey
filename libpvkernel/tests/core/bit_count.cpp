/**
 * \file bit_count.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVBitCount.h>
#include <iostream>

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

int main()
{
	int ret = 0;
#pragma omp parallel for
	for (uint32_t i = 0; i < 0xFFFFFFFFUL; i++) {
		size_t ref = bit_count_ref(i);
		size_t test = PVCore::PVBitCount::bit_count(i);
		if (ref != test) {
#pragma omp critical
			{
				std::cout << "bit count failed for 32-bit " << i << " : " << test << " vs. " << ref << " (ref)" << std::endl;
				ret = 1;
			}
		}
	}
	if (ret) {
		return ret;
	}

	std::cout << "32-bit tests done..." << std::endl;

#pragma omp parallel for
	for (uint64_t i = 0; i < 0xFFFFFFFFFFFFFFFFULL; i++) {
		size_t ref = bit_count_ref(i);
		size_t test = PVCore::PVBitCount::bit_count(i);
		if (ref != test) {
#pragma omp critical
			{
				std::cout << "bit count failed for 64-bit " << i << " : " << test << " vs. " << ref << " (ref)" << std::endl;
				ret = 1;
			}
		}
	}

	return ret;
}
