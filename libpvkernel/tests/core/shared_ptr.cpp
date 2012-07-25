/**
 * \file shared_ptr.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <iostream>
#include <cassert>
#include <sstream>

#include <pvkernel/core/PVSharedPointer.h>

struct pouet
{
	std::string print()
	{
		std::stringstream ss;
		ss << "pouet: " << this;
		return ss.str();
	}
};

typedef struct PVCore::PVSharedPtr<pouet> pouet_p;

template <typename T>
void deleter(T *p)
{
	std::cout << "    template deleter for address "
	          << p  << std::endl;
	delete p;
}

int main()
{
	std::cout << "##########################################" << std::endl;
	std::cout << "creation" << std::endl;
	std::cout << "  p_p1 = new pouet..." << std::endl;
	pouet_p p_p1(new pouet);

	assert(p_p1.use_count() == 1);
	std::cout << "    p_p1.use_count() -> " << p_p1.use_count() << std::endl;

	std::cout << "##########################################" << std::endl;
	std::cout << "test in sub scope" << std::endl;
	{
		std::cout << "  enter scope" << std::endl;
		pouet_p p_p2 = p_p1;

		std::cout << "    p_p2 = p_p1" << std::endl;
		assert(p_p1.use_count() == 2);
		assert(p_p2.use_count() == 2);
		std::cout << "    p_p1.use_count() -> " << p_p1.use_count() << std::endl;
		std::cout << "    p_p2.use_count() -> " << p_p2.use_count() << std::endl;
	}

	std::cout << "  leave scope" << std::endl;
	std::cout << "    p_p1.use_count() -> " << p_p1.use_count() << std::endl;

	std::cout << "##########################################" << std::endl;
	std::cout << "test of reset()" << std::endl;
	std::cout << "  p_p2 = p_p1" << std::endl;
	pouet_p p_p2 = p_p1;

	assert(p_p1.use_count() == 2);
	assert(p_p2.use_count() == 2);

	std::cout << "    p_p1.use_count() -> " << p_p1.use_count() << std::endl;
	std::cout << "  call to p_p2.reset()" << std::endl;
	p_p2.reset();

	assert(p_p1.use_count() == 1);
	assert(p_p2.use_count() == 1);

	std::cout << "    p_p1.use_count() -> " << p_p1.use_count() << std::endl;
	std::cout << "    p_p2.use_count() -> " << p_p2.use_count() << std::endl;

	std::cout << "##########################################" << std::endl;
	std::cout << "test of reset() in sub scope" << std::endl;
	{
		std::cout << "  enter scope" << std::endl;
		std::cout << "    p_p3 = p_p1" << std::endl;
		pouet_p p_p3 = p_p1;

		std::cout << "    p_p1.use_count() -> " << p_p1.use_count() << std::endl;
		std::cout << "    p_p3.use_count() -> " << p_p3.use_count() << std::endl;

		std::cout << "  call to p_p3.reset()" << std::endl;
		p_p3.reset();

		assert(p_p1.use_count() == 1);
		assert(p_p3.use_count() == 1);

		std::cout << "    p_p1.use_count() -> " << p_p1.use_count() << std::endl;
		std::cout << "    p_p3.use_count() -> " << p_p3.use_count() << std::endl;
	}

	std::cout << "  leave scope" << std::endl;
	std::cout << "    p_p1.use_count() -> " << p_p1.use_count() << std::endl;
	assert(p_p1.use_count() == 1);

	std::cout << std::boolalpha;

	std::cout << "##########################################" << std::endl;
	std::cout << "test of get()" << std::endl;
	std::cout << "  p_1 is set" << std::endl;
	std::cout << "  p_2 is reset" << std::endl;
	p_p2.reset();

	assert(p_p1.get() != nullptr);
	std::cout << "    p_1.get() -> " << p_p1.get() << std::endl;
	assert(p_p2.get() == nullptr);
	std::cout << "    p_2.get() -> " << p_p2.get() << std::endl;

	std::cout << "##########################################" << std::endl;
	std::cout << "test of operator->" << std::endl;

	std::cout << "    p_1->print() -> '" << p_p1->print() << "'" << std::endl;
	//std::cout << "    p_2->print() -> '" << p_p2->print() << "'" << std::endl;

	std::cout << "##########################################" << std::endl;
	std::cout << "test of operator*" << std::endl;

	std::cout << "    (*p_1).print() -> '" << (*p_p1).print() << "'" << std::endl;
	//std::cout << "    (*p_2).print() -> '" << (*p_p2).print() << "'" << std::endl;

	std::cout << "##########################################" << std::endl;
	std::cout << "test of operator bool" << std::endl;
	std::cout << "  p_1 is set" << std::endl;
	std::cout << "  p_2 is reset" << std::endl;
	p_p2.reset();

	assert(p_p1);
	std::cout << "    (bool)p_1 -> " << (bool)p_p1 << std::endl;
	assert(!p_p2);
	std::cout << "    (bool)p_2 -> " << (bool)p_p2 << std::endl;

	std::cout << "##########################################" << std::endl;
	std::cout << "test of comparison operator" << std::endl;

	std::cout << "  p_p2 = p_p1" << std::endl;
	p_p2 = p_p1;
	std::cout << "  p_p3 = new pouet..." << std::endl;
	pouet_p p_p3 = pouet_p(new pouet);

	std::cout << "  operator ==" << std::endl;
	assert(p_p1 == p_p2);
	std::cout << "    p_1 == p_p2 -> " << (p_p1 == p_p2) << std::endl;
	assert(p_p1 != p_p3);
	std::cout << "    p_1 == p_p3 -> " << (p_p1 == p_p3) << std::endl;

	std::cout << "  operator !=" << std::endl;
	assert(!(p_p1 != p_p2));
	std::cout << "    p_1 != p_p2 -> " << (p_p1 != p_p2) << std::endl;
	assert(!(p_p1 == p_p3));
	std::cout << "    p_1 != p_p3 -> " << (p_p1 != p_p3) << std::endl;

	std::cout << "  operator <" << std::endl;
	assert(!(p_p1 < p_p2));
	std::cout << "    p_1 < p_p2 -> " << (p_p1 < p_p2) << std::endl;
	std::cout << "    p_1 < p_p3 -> " << (p_p1 < p_p3) << std::endl;

	std::cout << "  operator <=" << std::endl;
	assert(p_p1 <= p_p2);
	std::cout << "    p_1 <= p_p2 -> " << (p_p1 <= p_p2) << std::endl;
	std::cout << "    p_1 <= p_p3 -> " << (p_p1 <= p_p3) << std::endl;

	std::cout << "  operator >" << std::endl;
	assert(!(p_p1 > p_p2));
	std::cout << "    p_1 > p_p2 -> " << (p_p1 > p_p2) << std::endl;
	std::cout << "    p_1 > p_p3 -> " << (p_p1 > p_p3) << std::endl;

	std::cout << "  operator >=" << std::endl;
	assert(p_p1 >= p_p2);
	std::cout << "    p_1 >= p_p2 -> " << (p_p1 >= p_p2) << std::endl;
	std::cout << "    p_1 >= p_p3 -> " << (p_p1 >= p_p3) << std::endl;

	std::cout << "##########################################" << std::endl;
	std::cout << "p_p1.reset() (needed for next steps)" << std::endl;
	p_p1.reset();

	std::cout << "##########################################" << std::endl;
	std::cout << "test of lambda deleter on one shared_ptr" << std::endl;
	pouet *p = new pouet;
	std::cout << "  p = " << p << std::endl;

	p_p2 = pouet_p(p);
	std::cout << "  p_p2 is set" << std::endl;
	std::cout << "  p_p2.set_deleter(...)" << std::endl;
	p_p2.set_deleter([&](pouet* p)
	                 {
		                 std::cout << "    lambda deleter for type pouet "
		                           << p << std::endl;
		                 delete p;
	                 });
	std::cout << "  p_p2.reset()" << std::endl;
	p_p2.reset();

	std::cout << "##########################################" << std::endl;
	std::cout << "test of template deleter on one shared_ptr" << std::endl;
	p = new pouet;
	std::cout << "  p = " << p << std::endl;

	p_p2 = pouet_p(p);
	std::cout << "  p_p2 is set" << std::endl;
	std::cout << "  p_p2.set_delete(...)" << std::endl;
	p_p2.set_deleter(deleter<pouet>);
	std::cout << "  p_p2.reset()" << std::endl;
	p_p2.reset();

	std::cout << "##########################################" << std::endl;
	std::cout << "test of deleter between two shared_ptr" << std::endl;
	std::cout << "  p_p3 is set" << std::endl;
	std::cout << "  p_p2 = p_p3" << std::endl;
	p_p2 = p_p3;

	std::cout << "  p_p2.set_deleter(...)" << std::endl;
	p_p2.set_deleter([&](pouet* p)
	                 {
		                 std::cout << "    lambda deleter for type pouet "
		                           << p << std::endl;
		                 delete p;
	                 });
	std::cout << "  p_p2.reset()" << std::endl;
	p_p2.reset();
	std::cout << "  p_p3.reset()" << std::endl;
	p_p3.reset();

	std::cout << "##########################################" << std::endl;
	std::cout << "test of make_shared(p)" << std::endl;

	p = new pouet;
	std::cout << "  p = " << p << std::endl;

	p_p2 = PVCore::make_shared<pouet>(p);
	std::cout << "  p_p2 is set" << std::endl;
	std::cout << "  (bool)p_p2 -> " << (bool)p_p2 << std::endl;
	std::cout << "  p_p2.get() -> " << p_p2.get() << std::endl;
	std::cout << "  p_p2.use_count() -> " << p_p2.use_count() << std::endl;
	std::cout << "  p_p2.reset()" << std::endl;
	p_p2.reset();

	std::cout << "##########################################" << std::endl;
	std::cout << "test of make_shared(p, lambda deleter)" << std::endl;

	p = new pouet;
	std::cout << "  p = " << p << std::endl;

	p_p2 = PVCore::make_shared<pouet>(p, [&](pouet* p)
	                                  {
		                                  std::cout << "    lambda deleter for type pouet "
		                                            << p  << std::endl;
		                                  delete p;
	                                  });
	std::cout << "  p_p2 is set" << std::endl;
	std::cout << "  (bool)p_p2 -> " << (bool)p_p2 << std::endl;
	std::cout << "  p_p2.get() -> " << p_p2.get() << std::endl;
	std::cout << "  p_p2.use_count() -> " << p_p2.use_count() << std::endl;
	std::cout << "  p_p2.reset()" << std::endl;
	p_p2.reset();

	std::cout << "##########################################" << std::endl;
	std::cout << "test of make_shared(p, template deleter)" << std::endl;

	p = new pouet;
	std::cout << "  p = " << p << std::endl;

	p_p2 = PVCore::make_shared<pouet>(p, deleter<pouet>);
	std::cout << "  p_p2 is set" << std::endl;
	std::cout << "  (bool)p_p2 -> " << (bool)p_p2 << std::endl;
	std::cout << "  p_p2.get() -> " << p_p2.get() << std::endl;
	std::cout << "  p_p2.use_count() -> " << p_p2.use_count() << std::endl;
	std::cout << "  p_p2.reset()" << std::endl;
	p_p2.reset();

	std::cout << "##########################################" << std::endl;
	std::cout << "end (there must be no text after)" << std::endl;

	return 0;
}
