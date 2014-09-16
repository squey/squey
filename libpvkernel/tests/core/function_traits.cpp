/**
 * \file function_traits.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <iostream>
#include <typeinfo>

#include <pvkernel/core/PVTypeTraits.h>
#include <pvkernel/core/PVFunctionTraits.h>

#include <boost/thread.hpp>

#include <pvkernel/core/picviz_assert.h>

void f(int a, short b, char c, uint64_t d)
{
	std::cout << a << " " << b << " " << (int) c << " " << d << std::endl;
}

void f2(int* a, const short* b, const short& c, short& d)
{
	std::cout << *a << " " << *b << c << d << std::endl;
}

void fsolo(int /*a*/)
{
}

void fnoarg()
{
}

struct A
{
	size_t f(size_t i) const { return i*_i; }
	void add(size_t& i) { i++; }
	size_t _i;
};
int main(int /*argc*/, char** /*argv*/)
{
	typedef PVCore::PVTypeTraits::function_traits<decltype(f)> ftraits;

	std::cout << ftraits::arity << std::endl;
	std::cout << typeid(ftraits::arguments_type::arg_type).name() << std::endl;
	std::cout << typeid(ftraits::arguments_type::next_arg_type::arg_type).name() << std::endl;

	std::cout << typeid(ftraits::type_of_arg<0>::type).name() << std::endl;
	std::cout << typeid(ftraits::type_of_arg<1>::type).name() << std::endl;

	std::cout << sizeof(ftraits::arguments_type) << std::endl;

	ftraits::arguments_deep_copy_type args;
	boost::thread th([=,&args] {
			ftraits::arguments_type my_args;
			my_args.set_args(1, 2, 4, 5);
			args = my_args;
			});
	th.join();
	std::cout << std::get<0>(args) << " " << std::get<1>(args) << " " << (int) std::get<2>(args) << " " << std::get<3>(args) << std::endl;

	typedef PVCore::PVTypeTraits::function_traits<decltype(f2)> ftraits2;

	ftraits2::arguments_type args2;
	int a = 2; short b = 4; const short d = 5;
	args2.set_args(&a, &b, d, b);
	std::cout << &a << " " << &b << std::endl;
	std::cout << std::get<0>(args2) << " " << std::get<1>(args2) << " " << std::get<2>(args2) << " " << std::get<3>(args2) << std::endl;

	typedef PVCore::PVTypeTraits::function_traits<decltype(fnoarg)> ftraits_noarg;
	ftraits_noarg::arguments_type noargs;
	noargs.set_args();

	{
		A a;
		a._i = 4;
		typedef PVCore::PVTypeTraits::function_traits<decltype(&A::f)> ftraits_af;
		ftraits_af::arguments_type args_af;
		args_af.set_args(6);
		PV_VALID(ftraits_af::call<&A::f>(a, args_af), (size_t)24, "a._i", 4, "args_af", (size_t)6);
	}

	{
		A a;
		size_t i = 4;
		typedef PVCore::PVTypeTraits::function_traits<decltype(&A::add)> ftraits_aadd;
		ftraits_aadd::arguments_deep_copy_type args;
		args.set_args(i);
		ftraits_aadd::call<&A::add>(a, args);
		PV_VALID(i, (size_t)5, "old i", (size_t)4);
	}

	return 0;
}
