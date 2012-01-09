#include <pvkernel/core/PVChunk.h>
#include <pvkernel/core/PVElement.h>
#include <pvkernel/core/PVField.h>

#include <iostream>

int main()
{
	std::cout << "sizeof PVChunk:" << sizeof(PVCore::PVChunk) << std::endl;
	std::cout << "sizeof PVElement:" << sizeof(PVCore::PVElement) << std::endl;
	std::cout << "sizeof PVField:" << sizeof(PVCore::PVField) << std::endl;
	std::cout << "sizeof PVBufferSlice:" << sizeof(PVCore::PVBufferSlice) << std::endl;
	std::cout << "sizeof QString:" << sizeof(QString) << std::endl;
}
