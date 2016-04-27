/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVChunk.h>
#include <pvkernel/core/PVElement.h>
#include <pvkernel/core/PVField.h>
#include <pvkernel/core/PVColor.h>

#include <type_traits>
#include <iostream>

int main()
{
	std::cout << "sizeof PVChunk:" << sizeof(PVCore::PVChunk) << std::endl;
	std::cout << "sizeof PVElement:" << sizeof(PVCore::PVElement) << std::endl;
	std::cout << "sizeof PVField:" << sizeof(PVCore::PVField) << std::endl;
	std::cout << "sizeof PVBufferSlice:" << sizeof(PVCore::PVBufferSlice) << std::endl;
	std::cout << "sizeof QString:" << sizeof(QString) << std::endl;
	std::cout << "sizeof PVColor:" << sizeof(PVCore::PVColor)
	          << ", is POD:" << std::is_pod<PVCore::PVColor>::value << std::endl;

	return 0;
}
