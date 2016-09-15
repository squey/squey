/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/inendi_assert.h>
#include <pvkernel/core/PVHugePODVector.h>

#include <iostream>

#define N (512UL * 1024 * 1024)
#define N2 (N / 2)

int main()
{
	typedef PVCore::PVHugePODVector<uint32_t, 16> pod_vector;

	pod_vector v;
	v.resize(N);
	PV_ASSERT_VALID((uintptr_t)(v.begin()) % 16 == 0);
	PV_VALID(v.size(), N);
	std::cout << "mmap done for 2GB." << std::endl;

	std::cout << "Write them." << std::endl;
	memset(v.begin(), 0x0A, N * sizeof(uint32_t));
	std::cout << "500 Mio written." << std::endl;

	std::cout << "Shrinking to 1GB." << std::endl;
	v.resize(N2);
	PV_VALID(v.size(), N2);
	std::cout << "Shrink done" << std::endl;

	std::cout << "Checking that values are correct." << std::endl;
	for (size_t i = 0; i < N2; i++) {
		PV_ASSERT_VALID(v.at(i) == 0x0A0A0A0AU, "i", i);
	}
	std::cout << "All valid" << std::endl;
	std::cout << "Clear the buffer." << std::endl;
	v.clear();
	PV_VALID(v.size(), (size_t)0);
	PV_VALID(v.begin(), (pod_vector::pointer) nullptr);

	return 0;
}
