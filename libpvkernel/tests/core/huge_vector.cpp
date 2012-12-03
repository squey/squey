#include <pvkernel/core/picviz_assert.h>
#include <pvkernel/core/PVHugePODVector.h>
#include <pvkernel/core/stdint.h>

#include <iostream>

#define N (4*1024UL*1024*1024)
#define N2 (N / 4)

int main()
{
	typedef PVCore::PVHugePODVector<uint32_t, 16> pod_vector;

	pod_vector v;
	v.resize(N);
	PV_ASSERT_VALID((uintptr_t)(v.begin()) % 16 == 0);
	PV_VALID(v.size(), N);
	std::cout << "mmap done for 16GB." << std::endl;

	std::cout << "Write them." << std::endl;
	memset(v.begin(), 0x0A, N*sizeof(uint32_t));
	std::cout << "4 Gio written." << std::endl;

	std::cout << "Shrinking to 1GB." << std::endl;
	v.resize(N2);
	PV_VALID(v.size(), N2);
	std::cout << "Shrink done" << std::endl;

	std::cout << "Checking that values are correct." << std::endl;
	for (size_t i = 0; i < N/4; i++) {
		PV_ASSERT_VALID(v.at(i) == 0x0A0A0A0AU, "i", i);
	}
	std::cout << "All valid" << std::endl;
	std::cout << "Clear the buffer." << std::endl;
	v.clear();
	PV_VALID(v.size(), (size_t)0);
	PV_VALID(v.begin(), (pod_vector::pointer)NULL);

	return 0;
}
