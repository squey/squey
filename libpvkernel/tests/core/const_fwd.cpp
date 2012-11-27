
#include <pvkernel/core/PVTypeTraits.h>
#include <iostream>

#include <pvkernel/core/picviz_assert.h>

struct A
{
	bool is_const() { return false; }
	bool is_const() const { return true; }
};

class B { };


template <typename T>
struct defined_with_const
{
	constexpr static bool value = false;
};

template <typename T>
struct defined_with_const<const T>
{
	constexpr static bool value = true;
};

template <typename T>
struct defined_with_const<const T*>
{
	constexpr static bool value = true;
};

template <typename T>
struct defined_with_const<const T&>
{
	constexpr static bool value = true;
};

template <typename T>
bool is_const(T&)
{
	return defined_with_const<T>::value;
}

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

	PV_VALID_P(is_const(gna10), true);

	return 0;
}
