#include <pvkernel/core/PVBitVisitor.h>
#include <iostream>

int main()
{
	uint64_t v = (1UL<<0) | (1UL<<4) | (1UL<<11) | (1UL<<18) | (1UL<<26) | (1UL<<35) | (1UL << 46) | (1UL<<59);
	PVCore::PVBitVisitor::visit_bits(v, [=](unsigned int b) { std::cout << b << std::endl; });

	return 0;
}
