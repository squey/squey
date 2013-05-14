/**
 * \file shared_ptr.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <iostream>
#include <cassert>
#include <sstream>

#include <pvkernel/core/PVSharedPointer.h>

#include <pvkernel/core/picviz_assert.h>

struct pouet
{
	pouet *get_addr()
	{
		return this;
	}
};

typedef struct PVCore::PVSharedPtr<pouet> pouet_p;

/* This code is required because GCC seems to distinguish a lambda function using
 * upper-scope variables and a lambda function which does not use upper-scope
 * variables.
 */
static bool deleted_status;

static void deleted_status_set()
{
	deleted_status = true;
}

// same as above
void deleter(void *p)
{
	(void)p;
	deleted_status = true;
}

int main()
{
	std::cout << "##########################################" << std::endl;
	std::cout << "creation" << std::endl;
	std::cout << "  p_p1 = new pouet..." << std::endl;
	pouet_p p_p1(new pouet);
	pouet_p p_p2;

	std::cout << "##########################################" << std::endl;
	std::cout << "test of ::get()" << std::endl;
	PV_VALID(p_p1.use_count(), 1L);
	PV_ASSERT_VALID(p_p1.get() != nullptr);

	PV_VALID(p_p2.use_count(), 1L);
	PV_ASSERT_VALID(p_p2.get() == nullptr);

	std::cout << "##########################################" << std::endl;
	std::cout << "test in sub scope" << std::endl;
	{
		std::cout << "  enter scope" << std::endl;
		pouet_p p_p2 = p_p1;
		std::cout << "    p_p2 = p_p1" << std::endl;

		PV_ASSERT_VALID(p_p2.get() == p_p1.get());
		PV_VALID(p_p1.use_count(), 2L);
		PV_VALID(p_p2.use_count(), 2L);
	}

	std::cout << "  leave scope" << std::endl;
	PV_VALID(p_p1.use_count(), 1L);

	std::cout << "##########################################" << std::endl;
	std::cout << "test of reset()" << std::endl;
	std::cout << "  p_p2 = p_p1" << std::endl;
	p_p2 = p_p1;

	std::cout << "  call to p_p2.reset()" << std::endl;
	p_p2.reset();

	PV_ASSERT_VALID(p_p2.get() != p_p1.get());
	PV_VALID(p_p1.use_count(), 1L);
	PV_VALID(p_p2.use_count(), 1L);

	std::cout << "##########################################" << std::endl;
	std::cout << "test of reset() in sub scope" << std::endl;
	{
		std::cout << "  enter scope" << std::endl;
		std::cout << "    p_p3 = p_p1" << std::endl;
		pouet_p p_p3 = p_p1;

		PV_ASSERT_VALID(p_p3.get() == p_p1.get());
		PV_VALID(p_p1.use_count(), 2L);
		PV_VALID(p_p3.use_count(), 2L);

		std::cout << "  call to p_p3.reset()" << std::endl;
		p_p3.reset();

		PV_ASSERT_VALID(p_p3.get() != p_p1.get());
		PV_VALID(p_p1.use_count(), 1L);
		PV_VALID(p_p3.use_count(), 1L);
	}

	std::cout << "  leave scope" << std::endl;
	PV_VALID(p_p1.use_count(), 1L);

	std::cout << std::boolalpha;

	std::cout << "##########################################" << std::endl;
	std::cout << "test of operator->" << std::endl;

	PV_VALID(p_p1->get_addr(), p_p1.get());
	PV_VALID(p_p2->get_addr(), p_p2.get());

	std::cout << "##########################################" << std::endl;
	std::cout << "test of operator*" << std::endl;

	PV_VALID((*p_p1).get_addr(), p_p1.get());
	PV_VALID((*p_p2).get_addr(), p_p2.get());

	std::cout << "##########################################" << std::endl;
	std::cout << "test of operator bool" << std::endl;
	p_p2.reset();

	PV_VALID((bool)p_p1, true);
	PV_VALID((bool)p_p2, false);

	std::cout << "##########################################" << std::endl;
	std::cout << "test of comparison operator" << std::endl;

	std::cout << "  p_p2 = p_p1" << std::endl;
	p_p2 = p_p1;
	std::cout << "  p_p3 = new pouet..." << std::endl;
	pouet_p p_p3 = pouet_p(new pouet);

	std::cout << "  operator ==" << std::endl;
	PV_VALID(p_p1 == p_p2, true);
	PV_VALID(p_p1 == p_p3, false);

	std::cout << "  operator !=" << std::endl;

	PV_VALID(p_p1 != p_p2, false);
	PV_VALID(p_p1 != p_p3, true);

	std::cout << "  operator <" << std::endl;

	PV_VALID(p_p1 < p_p2, false);
	PV_VALID(p_p1 < p_p3, true);

	std::cout << "  operator <=" << std::endl;

	PV_VALID(p_p1 <= p_p2, true);
	PV_VALID(p_p1 <= p_p3, true);

	std::cout << "  operator >" << std::endl;
	PV_VALID(p_p1 > p_p2, false);
	PV_VALID(p_p1 > p_p3, false);

	std::cout << "  operator >=" << std::endl;
	PV_VALID(p_p1 >= p_p2, true);
	PV_VALID(p_p1 >= p_p3, false);

	pouet *p;

	std::cout << "##########################################" << std::endl;
	std::cout << "test of lambda deleter on one shared_ptr" << std::endl;
	p = new pouet;

	p_p2 = pouet_p(p);
	std::cout << "  p_p2 is set" << std::endl;
	std::cout << "  p_p2.set_deleter(...)" << std::endl;
	p_p2.set_deleter([](void*)
	                 {
		                 deleted_status_set();
	                 });
	deleted_status = false;
	std::cout << "  p_p2.reset()" << std::endl;
	p_p2.reset();
	PV_VALID(deleted_status, true);

	std::cout << "##########################################" << std::endl;
	std::cout << "test of template deleter on one shared_ptr" << std::endl;
	p = new pouet;

	p_p2 = pouet_p(p);
	std::cout << "  p_p2 is set" << std::endl;
	std::cout << "  p_p2.set_delete(...)" << std::endl;
	p_p2.set_deleter(deleter);

	deleted_status = false;
	std::cout << "  p_p2.reset()" << std::endl;
	p_p2.reset();
	PV_VALID(deleted_status, true);

	std::cout << "##########################################" << std::endl;
	std::cout << "test of deleter between two shared_ptr" << std::endl;
	std::cout << "  p_p3 is set" << std::endl;
	std::cout << "  p_p2 = p_p3" << std::endl;
	p_p2 = p_p3;

	std::cout << "  p_p2.set_deleter(...)" << std::endl;
	p_p2.set_deleter([](void*)
	                 {
		                 deleted_status_set();
	                 });

	deleted_status = false;
	std::cout << "  p_p2.reset()" << std::endl;
	p_p2.reset();
	PV_VALID(deleted_status, false);
	std::cout << "  p_p3.reset()" << std::endl;
	p_p3.reset();
	PV_VALID(deleted_status, true);

	std::cout << "##########################################" << std::endl;
	std::cout << "test of make_shared(p)" << std::endl;

	p = new pouet;

	p_p2 = PVCore::make_shared<pouet>(p);

	PV_VALID((bool)p_p2, true);
	PV_VALID(p_p2.get(), p);
	PV_VALID(p_p2.use_count(), 1L);
	p_p2.reset();

	std::cout << "##########################################" << std::endl;
	std::cout << "test of make_shared(p, lambda deleter)" << std::endl;

	p = new pouet;

	p_p2 = PVCore::make_shared<pouet>(p, [](void*)
	                                  {
		                                  deleted_status_set();
	                                  });

	PV_VALID((bool)p_p2, true);
	PV_VALID(p_p2.get(), p);
	PV_VALID(p_p2.use_count(), 1L);
	deleted_status = false;
	std::cout << "  p_p2.reset()" << std::endl;
	p_p2.reset();
	PV_VALID(deleted_status, true);

	std::cout << "##########################################" << std::endl;
	std::cout << "test of make_shared(p, template deleter)" << std::endl;

	p = new pouet;

	p_p2 = PVCore::make_shared<pouet>(p, deleter);

	PV_VALID((bool)p_p2, true);
	PV_VALID(p_p2.get(), p);
	PV_VALID(p_p2.use_count(), 1L);
	deleted_status = false;
	std::cout << "  p_p2.reset()" << std::endl;
	p_p2.reset();
	PV_VALID(deleted_status, true);

	std::cout << "##########################################" << std::endl;
	std::cout << "end (there must be no text after)" << std::endl;

	return 0;
}
