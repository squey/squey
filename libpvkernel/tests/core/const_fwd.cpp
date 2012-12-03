
#include <pvkernel/core/PVTypeTraits.h>
#include <iostream>

#include <pvkernel/core/picviz_assert.h>

#include "is_const.h"

struct A
{
	bool is_const() { return false; }
	bool is_const() const { return true; }

	operator std::string () const
	{
		std::stringstream ret;
		ret << std::string("class A(") << this << ")";
		return ret.str();
	}
};


std::ostream& operator << (std::ostream& o, const A& a)
{
	o << std::string(a);
	return o;
}


class B { };

int main()
{
	A a;

	typename PVCore::PVTypeTraits::const_fwd<A, B>::type gna;
	typename PVCore::PVTypeTraits::const_fwd<A, const B>::type gna2;

	typename PVCore::PVTypeTraits::const_fwd<void, B>::type* gna4;
	typename PVCore::PVTypeTraits::const_fwd<void, const B>::type* gna8;

	typename PVCore::PVTypeTraits::const_fwd<A, B>::type& gnaa = a;
	typename PVCore::PVTypeTraits::const_fwd<A, const B>::type& gna10 = a;

	PV_VALID_P(gna.is_const(), false);

	PV_VALID_P(gna2.is_const(), true);

	PV_VALID_P(is_const(gna4), false);

	PV_VALID_P(is_const(gna8), true);

	PV_VALID_P(is_const(gnaa), false);

	PV_VALID_P(is_const(gna10), false, "gna10", gna10, "i", 10);

	return 0;
}
