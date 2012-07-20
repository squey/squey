#include <iostream>
#include <typeinfo>

#include <pvkernel/core/PVTypeTraits.h>
#include <pvkernel/core/PVFunctionTraits.h>

#include <boost/type_traits.hpp>

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

int main(int /*argc*/, char** /*argv*/)
{
	typedef PVCore::PVTypeTraits::function_traits<decltype(f)> ftraits;

	std::cout << ftraits::arity << std::endl;
	std::cout << typeid(ftraits::arguments_type::arg_type).name() << std::endl;
	std::cout << typeid(ftraits::arguments_type::next_arg_type::arg_type).name() << std::endl;

	std::cout << typeid(ftraits::type_of_arg<0>::type).name() << std::endl;
	std::cout << typeid(ftraits::type_of_arg<1>::type).name() << std::endl;

	std::cout << sizeof(ftraits::arguments_type) << std::endl;

	ftraits::arguments_type args;
	args.set_args(1, 2, 4, 5);
	std::cout << args.get_arg<0>() << " " << args.get_arg<1>() << " " << (int) args.get_arg<2>() << " " << args.get_arg<3>() << std::endl;

	typedef PVCore::PVTypeTraits::function_traits<decltype(f2)> ftraits2;

	ftraits2::arguments_type args2;
	int a = 2; short b = 4; const short d = 5;
	args2.set_args(&a, &b, d, b);
	std::cout << &a << " " << &b << std::endl;
	std::cout << args2.get_arg<0>() << " " << args2.get_arg<1>() << " " << args2.get_arg<2>() << " " << args2.get_arg<3>() << std::endl;

	typedef PVCore::PVTypeTraits::function_traits<decltype(fnoarg)> ftraits_noarg;
	ftraits_noarg::arguments_type noargs;
	noargs.set_args();

	return 0;
}
