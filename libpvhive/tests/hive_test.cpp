/**
 * \file hive_test.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <iostream>

#include <pvhive/PVHive.h>
#include <pvhive/PVActor.h>
#include <pvhive/PVObserver.h>

class Ci
{
public:
	Ci(int i) : _i(i) {}

	void set_i(int i) { _i = i; }

	int get_i() const { return _i; }

private:
	int _i;
};

typedef PVHive::PVActor<Ci> CiActor;

class CiObs : public PVHive::PVObserver<Ci>
{
public:
	CiObs()
	{}

	virtual void refresh()
	{
		std::cout << "Obs::refresh(): " << get_object()->get_i() << std::endl;
	}

	virtual void about_to_be_deleted()
	{
		std::cout << "Obs::about_to_be_deleted()" << std::endl;
	}
};


int main()
{
	PVHive::PVHive &h1 = PVHive::PVHive::get();
	PVHive::PVHive &h2 = PVHive::PVHive::get();


	if (&h1 != &h2) {
		std::cerr << "PVHive::get() returns different addresses" << std::endl;
		return 1;
	}

	std::cout << "PVHive::get() works" << std::endl;

	Ci i(0);
	CiActor a;

	PVHive::PVHive::get().register_actor(i, a);

	CiObs o;

	PVHive::PVHive::get().register_observer(i, o);

	a.call<decltype(&Ci::set_i), &Ci::set_i>(24);

	PVACTOR_CALL(a, &Ci::set_i, 42);

	return 0;
}
