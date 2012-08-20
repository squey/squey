#include <pvkernel/core/PVTypeTraits.h>
#include <iostream>

struct A
{
	bool is_const() { return false; }
	bool is_const() const { return true; }
};

class B { };

int main()
{
	typename PVCore::PVTypeTraits::const_fwd<A, B>::type gna;
	typename PVCore::PVTypeTraits::const_fwd<A, const B>::type gna2;
	typename PVCore::PVTypeTraits::const_fwd<void, const B>::type* gna4;

	std::cout << gna.is_const() << std::endl;
	std::cout << gna2.is_const() << std::endl;

	return 0;
}
