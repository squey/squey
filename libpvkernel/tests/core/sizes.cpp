#include <pvkernel/core/PVChunk.h>
#include <pvkernel/core/PVElement.h>
#include <pvkernel/core/PVField.h>
#include <pvkernel/core/PVUnicodeString.h>
#include <pvkernel/core/PVColor.h>
#include <boost/type_traits.hpp>

#include <iostream>

int main()
{
	std::cout << "sizeof PVChunk:" << sizeof(PVCore::PVChunk) << std::endl;
	std::cout << "sizeof PVElement:" << sizeof(PVCore::PVElement) << std::endl;
	std::cout << "sizeof PVField:" << sizeof(PVCore::PVField) << std::endl;
	std::cout << "sizeof PVBufferSlice:" << sizeof(PVCore::PVBufferSlice) << std::endl;
	std::cout << "sizeof PVUnicodeString:" << sizeof(PVCore::PVUnicodeString) << std::endl;
	std::cout << "sizeof QString:" << sizeof(QString) << std::endl;
	std::cout << "sizeof PVColor:" << sizeof(PVCore::PVColor) << ", is POD:" << boost::is_pod<PVCore::PVColor>::value << std::endl;
}
