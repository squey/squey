#include <pvkernel/core/picviz_assert.h>
#include <pvkernel/core/PVHugePODVector.h>
#include <pvkernel/core/stdint.h>

#include <iostream>

#define N (4*1024ULL*1024*1024)

int main()
{
	PVCore::PVHugePODVector<uint32_t, 16> v;
	v.resize(N);
	ASSERT_VALID((uintptr_t)(v.begin()) % 16 == 0);
	std::cout << "mmap done for 16GB." << std::endl;
	std::cout << "Press a key to write them!" << std::endl;
	getchar();
	memset(v.begin(), 0x0A, N*sizeof(uint32_t));
	std::cout << "4GB written. Press a key to shrink to 1GB" << std::endl;
	getchar();
	v.resize(N/4);
	std::cout << "Shrink done. Press enter to check that values are correct." << std::endl;
	getchar();
	for (size_t i = 0; i < N/4; i++) {
		ASSERT_VALID(v.at(i) == 0x0A0A0A0AU);
	}
	std::cout << "All valid. Press enter to clear the buffer." << std::endl;
	getchar();
	v.clear();
	std::cout << "Buffer cleared. Press enter to exit." << std::endl;
	getchar();

	return 0;
}
