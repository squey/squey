/**
 * \file simplistic.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <iostream>

#include <QApplication>

#include <pvkernel/core/PVSharedPointer.h>

#include <pvhive/PVHive.h>
#include <pvhive/PVActor.h>
#include <pvhive/PVObserver.h>


struct Obj1
{
	~Obj1()
	{
		std::cout << "  Obj1::~Obj1 for object " << this << std::endl;
	}

	void print()
	{
		std::cout << "  Obj1::print" << std::endl;
	}
};

typedef PVCore::PVSharedPtr<Obj1> Obj1_p;

struct Obj2
{
	~Obj2()
	{
		std::cout << "  Obj2::~Obj2 for object " << this << std::endl;
	}

	void print()
	{
		std::cout << "  Obj2::print" << std::endl;
	}

	void my_action(int param)
	{
		std::cout << "  Obj2::myaction with " << param << std::endl;
	}

	void test_ref(int& i)
	{
		i += 1;
		std::cout << "  Obj2::test_ref with i = " << i << " (" << &i << ")" << std::endl;
	}
};

typedef PVCore::PVSharedPtr<Obj2> Obj2_p;

class Obj1Observer : public PVHive::PVObserver<Obj1>
{
public:
	virtual void refresh()
	{
		std::cout << "  Obj1Observer::refresh for object " << get_object() << std::endl;
	}

	virtual void about_to_be_deleted()
	{
		std::cout << "    Obj1Observer::about_to_be_deleted for object " << get_object() << std::endl;
	}

private:
};

class Obj2Observer : public PVHive::PVObserver<Obj2>
{
public:
	virtual void refresh()
	{
		std::cout << "  Obj2Observer::refresh for object " << get_object() << std::endl;
	}

	virtual void about_to_be_deleted()
	{
		std::cout << "    Obj2Observer::about_to_be_deleted for object " << get_object() << std::endl;
	}

private:
};

class FuncObserver: public PVHive::PVFuncObserver<Obj2, decltype(&Obj2::my_action), &Obj2::my_action>
{
protected:
	virtual void update(arguments_type const& args) const
	{
		std::cout << "    FuncObserver on Obj2::my_action for object " << get_object() <<  "with param " << std::get<0>(args) << std::endl;
	}
};

int main(int argc, char **argv)
{
	QApplication app(argc, argv);

	PVHive::PVHive &h1 = PVHive::PVHive::get();
	PVHive::PVHive &h2 = PVHive::PVHive::get();

	if (&h1 != &h2) {
		std::cerr << "PVHive::get() returns different addresses" << std::endl;
		return 1;
	}

	std::cout << "# scene creation" << std::endl;

	std::cout << "#   hive " << &h1 << std::endl;

	Obj1_p o1 = Obj1_p(new Obj1);
	PVHive::PVHive::get().register_object(o1);

	PVHive::PVActor<Obj1> a1o1;
	PVHive::PVHive::get().register_actor(o1, a1o1);

	Obj2_p o2 = Obj2_p(new Obj2);
	PVHive::PVHive::get().register_object(o2);

	PVHive::PVActor<Obj2> a1o2;
	PVHive::PVHive::get().register_actor(o2, a1o2);

	Obj1Observer o1o1;
	PVHive::PVHive::get().register_observer(o1, o1o1);

	Obj2Observer o1o2;
	PVHive::PVHive::get().register_observer(o2, o1o2);

	Obj2Observer o2o2;
	PVHive::PVHive::get().register_observer(o2, o2o2);

	FuncObserver o2fo;
	PVHive::PVHive::get().register_func_observer(o2, o2fo);

	std::cout << "# a1o1 calls &Obj1::print" << std::endl;
	PVACTOR_CALL(a1o1, &Obj1::print);
	std::cout << "# a1o2 calls &Obj2::print" << std::endl;
	PVACTOR_CALL(a1o2, &Obj2::print);
	std::cout << "# a1o2 calls &Obj2::my_action" << std::endl;
	PVACTOR_CALL(a1o2, &Obj2::my_action, 1);
	int i = 4;
	std::cout << "# a1o2 calls &Obj2::test_ref with &i=" << &i << std::endl;
	PVACTOR_CALL(a1o2, &Obj2::test_ref, i);

	std::cout << "# unregister a1o1" << std::endl;
	PVHive::PVHive::get().unregister_actor(a1o1);

	std::cout << "# unregister o1 (and deletion)" << std::endl;

	std::cout << "- before unregister" << std::endl;
	PVHive::PVHive::get().print();
	o1.reset();
	std::cout << "- after unregister" << std::endl;
	PVHive::PVHive::get().print();

	std::cout << "# unregister o1o2" << std::endl;
	PVHive::PVHive::get().unregister_observer(o2o2);

	std::cout << "# end. obj2 should be deleted now" << std::endl;

	return 0;
}
