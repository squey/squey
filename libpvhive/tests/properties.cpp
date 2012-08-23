/**
 * \file properties.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVSharedPointer.h>

#include <pvhive/PVActor.h>
#include <pvhive/PVCallHelper.h>
#include <pvhive/PVHive.h>
#include <pvhive/PVObserver.h>

#include <boost/ref.hpp>
#include <boost/thread.hpp>

#include <stdlib.h>

class MyObjectProperty
{
public:
	MyObjectProperty(int v = 0): _v(v)
	{ }

	int get_value() { return _v; }
	void set_value(int v) { _v = v; }

public:
	int _v;
};

class MyObject
{
public:
	MyObject(int i): _i(i)
	{
		std::cout << "MyObject::_i: " << &_i << std::endl;
		std::cout << "MyObject::_prop: " << &_prop << std::endl;
	}

public:
	int const& get_i() const { return _i; };
	void set_i(int const& i) { _i = i; }
	void set_i2(int const& i) { _i = i; }

	void set_prop(MyObjectProperty const& p) { _prop = p; }
	MyObjectProperty const& get_prop() const { return _prop; }

private:
	int _i;
	MyObjectProperty _prop;
};

typedef PVCore::PVSharedPtr<MyObject> MyObject_p;

class MyObjectObserver : public PVHive::PVObserver<MyObjectProperty>
{
public:
	MyObjectObserver() {}

	void refresh()
	{
		std::cout << "  MyObjectObserver::refresh for object "
		          << get_object() << std::endl;
	}

	void about_to_be_deleted()
	{
		std::cout << "  MyObjectObserver::about_to_be_deleted for object "
		          << get_object() << std::endl;
	}
};


class MyObjectPropertyObserver : public PVHive::PVObserver<MyObjectProperty>
{
public:
	MyObjectPropertyObserver() {}

	void refresh()
	{
		std::cout << "  MyObjectPropertyObserver::refresh for object "
		          << get_object() << std::endl;
	}

	void about_to_be_deleted()
	{
		std::cout << "  MyObjectPropertyObserver::about_to_be_deleted for object "
		          << get_object() << std::endl;
	}
};

PVHIVE_CALL_OBJECT_BLOCK_BEGIN()

/*template <>
void PVHive::PVHive::call_object<MyObject, decltype(&MyObject::set_prop), &MyObject::set_prop,
                                 boost::reference_wrapper<MyObjectProperty const> >(MyObject* o, boost::reference_wrapper<MyObjectProperty const> p)
{
	std::cout << "  PVHive::call_object for MyObject::set_prop" << std::endl;
	std::cout << "    in thread " << boost::this_thread::get_id() << std::endl;
	call_object_default<MyObject, decltype(&MyObject::set_prop), &MyObject::set_prop, MyObjectProperty const&>(o, p);
	refresh_observers(&o->get_prop());
}*/

template <>
void PVHive::PVHive::call_object<FUNC(MyObject::set_prop)>(MyObject* o, PVCore::PVTypeTraits::function_traits<decltype(&MyObject::set_prop)>::arguments_type const& args)
{
	std::cout << "  PVHive::call_object for MyObject::set_prop" << std::endl;
	std::cout << "    in thread " << boost::this_thread::get_id() << std::endl;
	call_object_default<MyObject, FUNC(MyObject::set_prop)>(o, args);
	refresh_observers(&o->get_prop());
}

PVHIVE_CALL_OBJECT_BLOCK_END()


void hive_dump_content()
{
	std::cout << "# atexit: dumping hive content" << std::endl;
	PVHive::PVHive::get().print();
}

int main()
{
	PVHive::PVHive &hive = PVHive::PVHive::get();

	atexit(hive_dump_content);

	MyObject_p myobj = MyObject_p(new MyObject(42));

	hive.register_object(myobj);
	hive.register_object(myobj, [](MyObject const &myo) -> const MyObjectProperty &
	                     {
		                     return myo.get_prop();
	                     });


	// 1 acteur sur myobj
	PVHive::PVActor<MyObject> oactor;

	hive.register_actor(myobj, oactor);

	// 1 acteur sur myobj.prop
	PVHive::PVActor<MyObject> pactor;

	hive.register_actor(myobj, pactor);

	// 1 observeur sur myobj
	MyObjectObserver oobserver;
	std::cout << "# register observer " << &oobserver
	          << "#  for myobj " << &myobj << std::endl;
	hive.register_observer(myobj, oobserver);

	// 2 observeur sur myobj.prop, 1 en lambda et un en methode
	MyObjectPropertyObserver pobserver;
	std::cout << "# register observer " << &pobserver
	          << "#  for myobj.get_prop() " << &(myobj->get_prop()) << std::endl;
	hive.register_observer(myobj, [](MyObject const &myo) -> const MyObjectProperty &
	                       {
		                       return myo.get_prop();
	                       }, pobserver);

	std::cout << "# oactor.call(8)" << std::endl;
	PVACTOR_CALL(oactor, &MyObject::set_i, 8);

	std::cout << "# pactor.call(42)" << std::endl;
	PVACTOR_CALL(pactor, &MyObject::set_prop, boost::cref(MyObjectProperty(42)));

	std::cout << "# hive.unregister_object(myobj)" << std::endl;
	std::cout << "# end.";
	std::cout << "# dumping hive.";
	hive.print();

	std::cout << "# all myobj's observer must be unregistered after this message." << std::endl;

	return 0;
}






