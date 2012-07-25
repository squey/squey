/**
 * \file test_adrien_main-boost_thread.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <iostream>
#include <QApplication>

#include <pvhive/PVHive.h>
#include <pvhive/PVActor.h>
#include <pvhive/PVObserverCallback.h>

#include "adrien_objs.h"
#include "adrien_dlg.h"
#include "adrien_hdr.h"

#include <boost/thread.hpp>
#include <boost/bind.hpp>


PVHIVE_CALL_OBJECT_BLOCK_BEGIN()

template <>
void PVHive::PVHive::call_object<MyObject, decltype(&MyObject::set_prop), &MyObject::set_prop, boost::reference_wrapper<ObjectProperty const> >(MyObject* o, boost::reference_wrapper<ObjectProperty const> p)
{
	std::cout << "  PVHive::call_object for MyObject::set_prop" << std::endl;
	std::cout << "    in thread " << boost::this_thread::get_id() << std::endl;
	call_object_default<MyObject, decltype(&MyObject::set_prop), &MyObject::set_prop, ObjectProperty const&>(o, p);
	refresh_observers(&o->get_prop());
}

template <>
void PVHive::PVHive::call_object<MyObject, decltype(&MyObject::set_prop), &MyObject::set_prop, ObjectProperty>(MyObject* o, ObjectProperty p)
{
	std::cout << "  PVHive::call_object for MyObject::set_prop, non const& version" << std::endl;
	std::cout << "    in thread " << boost::this_thread::get_id() << std::endl;
	call_object_default<MyObject, decltype(&MyObject::set_prop), &MyObject::set_prop, ObjectProperty>(o, p);
	refresh_observers(&o->get_prop());
}

template <>
void PVHive::PVHive::call_object<MyObject, decltype(&MyObject::set_i), &MyObject::set_i, int>(MyObject* o, int i)
{
	std::cout << "  PVHive::call_object for MyObject::set_i" << std::endl;
	std::cout << "    in thread " << boost::this_thread::get_id() << std::endl;
	call_object_default<MyObject, decltype(&MyObject::set_i), &MyObject::set_i, int>(o, i);
}

template <>
void PVHive::PVHive::call_object<MyObject, decltype(&MyObject::set_i2), &MyObject::set_i2, int>(MyObject* o, int i)
{
	std::cout << "  PVHive::call_object for MyObject::set_i2" << std::endl;
	std::cout << "    in thread " << boost::this_thread::get_id() << std::endl;
	call_object_default<MyObject, decltype(&MyObject::set_i2), &MyObject::set_i2, int>(o, i);
}

PVHIVE_CALL_OBJECT_BLOCK_END()

/* would be nice to have:

a "class_func" class whose type defines completely a emberfunction of a class (see boost::function_traits that can help !)
a helper function "func" that creates class_func given a pointer-to-member function

template <>
void Hive::call_object<decltype(func(&MyObject::set_prop))>(MyObject* o, boost::reference_wrapper<ObjectProperty const> p)
{
	// etc...
}
*/

void update_prop(PVHive::PVHive& cc, MyObject_p& o)
{
	MyObjActor actor;
	cc.register_actor(o, actor);

	std::cout << "Update thread is " << boost::this_thread::get_id() << std::endl;
	int v = 0;
	while(true) {
		sleep(1);
		std::cout << "Update prop to " << v << std::endl;
		actor.call<decltype(&MyObject::set_prop), &MyObject::set_prop>(boost::cref(ObjectProperty(v)));
		v++;
	}
}

int main(int argc, char** argv)
{
	MyObject_p o = MyObject_p(new MyObject(4));
	MyObjActor actor;
	MyObjObserver observer;

	auto observer_callback = PVHive::create_observer_callback<MyObject>(
			[](MyObject const* o) { std::cout << "  Callback about_to_be_refreshed to i=" << o->get_i() << std::endl; },
			[](MyObject const* o) { std::cout << "  Callback refresh to i=" << o->get_i() << std::endl; },
			[](MyObject const* o) { std::cout << "  Callback delete for i=" << o->get_i() << std::endl; }
		);

	QApplication app(argc, argv);

	std::cout << "Main thread is " << boost::this_thread::get_id() << std::endl;

	PVHive::PVHive &hive = PVHive::PVHive::get();

	hive.register_object(o);
	hive.register_actor(o, actor);
	hive.register_observer(o, observer);
	hive.register_observer(o, observer_callback);

	actor.call<decltype(&MyObject::set_i), &MyObject::set_i>(8);
	actor.call<decltype(&MyObject::set_i2), &MyObject::set_i2>(9);

	TestDlg* dlg = new TestDlg(o, NULL);
	dlg->show();

	boost::thread th(boost::bind(update_prop, boost::ref(hive), boost::ref(o)));

	app.exec();

	return 0;
}
