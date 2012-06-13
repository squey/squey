#include <iostream>
#include <QApplication>

#include <pvhive/PVHive.h>
#include <pvhive/PVActor.h>
#include <pvhive/PVObserver.h>
#include <pvhive/PVObserverCallback.h>

#include "test_adrien_objs.h"
#include "test_adrien_dlg.h"

#include <boost/thread.hpp>
#include <boost/bind.hpp>

class MyObjObserver: public PVHive::PVObserver<MyObject>
{
public:
	void refresh() { std::cout << "refresh obj observer: " << get_object()->get_i() << std::endl; }
	void about_to_be_deleted() { }
};

class MyObjActor: public PVHive::PVActor<MyObject>
{
public:
#if 0
	template <typename F, F f, typename... Ttypes>
	void call(Ttypes... params)
	{
		Actor<MyObject>::call<F, f>(params...);
		/*if (f == &MyObject::set_i2) {
			std::cout << "actor int2 custom" << std::endl;
		}
		else
		if (f == &MyObject::set_prop) {
			parent_cc().refresh_observers(&_p._prop);
		}*/
	}
#endif

	/*
	template <typename... Ttypes>
	void call(decltype(&MyObject::set_i2) f, Ttypes... params)
	{
		std::cout << "set_i2 special actor" << std::endl;
		Actor<MyObject>::call(f, params...);
	}

	template <typename... Ttypes>
	void call(decltype(&MyObject::set_i) f, Ttypes... params)
	{
		std::cout << "set_i special actor" << std::endl;
		Actor<MyObject>::call(f, params...);
	}*/
};

namespace PVHive
{

template <>
void PVHive::PVHive::call_object<MyObject, decltype(&MyObject::set_prop), &MyObject::set_prop, boost::reference_wrapper<ObjectProperty const> >(MyObject* o, boost::reference_wrapper<ObjectProperty const> p)
{
	std::cout << "specialized control center call for MyObject::set_prop" << std::endl;
	std::cout << "Hive call in thread " << boost::this_thread::get_id() << std::endl;
	call_object_default<MyObject, decltype(&MyObject::set_prop), &MyObject::set_prop, ObjectProperty const&>(o, p);
	refresh_observers(&o->get_prop());
}

template <>
void PVHive::PVHive::call_object<MyObject, decltype(&MyObject::set_prop), &MyObject::set_prop, ObjectProperty>(MyObject* o, ObjectProperty p)
{
	std::cout << "specialized control center call for MyObject::set_prop, non const& version" << std::endl;
	std::cout << "Hive call in thread " << boost::this_thread::get_id() << std::endl;
	call_object_default<MyObject, decltype(&MyObject::set_prop), &MyObject::set_prop, ObjectProperty>(o, p);
	refresh_observers(&o->get_prop());
}

template <>
void PVHive::PVHive::call_object<MyObject, decltype(&MyObject::set_i), &MyObject::set_i, int>(MyObject* o, int i)
{
	std::cout << "specialized control center call for MyObject::set_i" << std::endl;
	call_object_default<MyObject, decltype(&MyObject::set_i), &MyObject::set_i, int>(o, i);
}

template <>
void PVHive::PVHive::call_object<MyObject, decltype(&MyObject::set_i2), &MyObject::set_i2, int>(MyObject* o, int i)
{
	std::cout << "specialized control center call for MyObject::set_i2" << std::endl;
	call_object_default<MyObject, decltype(&MyObject::set_i2), &MyObject::set_i2, int>(o, i);
}

}

/* would be nice to have:

a "class_func" class whose type defines completely a emberfunction of a class (see boost::function_traits that can help !)
a helper function "func" that creates class_func given a pointer-to-member function

template <>
void Hive::call_object<decltype(func(&MyObject::set_prop))>(MyObject* o, boost::reference_wrapper<ObjectProperty const> p)
{
	// etc...
}
*/

void update_prop(PVHive::PVHive& cc, MyObject& o)
{
	MyObjActor actor;
	cc.register_actor(o, actor);

	int v = 0;
	while(true) {
		sleep(1);
		actor.call<decltype(&MyObject::set_prop), &MyObject::set_prop>(boost::cref(ObjectProperty(v)));
		//actor.easy_call(func(&MyObject::set_prop), boost::cref(ObjectProperty(v)));
		v++;
	}
}

int main(int argc, char** argv)
{
	MyObject o(4);
	MyObjActor actor;
	MyObjObserver observer;

	auto observer_callback = PVHive::create_observer_callback<MyObject>(
			[](MyObject const* o) { std::cout << "refresh obj observer lambda: " << o->get_i() << std::endl; },
			[](MyObject const* o) { std::cout << "delete lambda: " << o->get_i() << std::endl; }
		);

	PVHive::PVHive &hive = PVHive::PVHive::get();
	hive.register_actor(o, actor);
	hive.register_observer(o, observer);
	hive.register_observer(o, observer_callback);

	actor.call<decltype(&MyObject::set_i), &MyObject::set_i>(8);
	actor.call<decltype(&MyObject::set_i2), &MyObject::set_i2>(9);

	QApplication app(argc, argv);

	TestDlg* dlg = new TestDlg(hive, o, NULL);
	dlg->show();

	std::cout << "Main thread is " << boost::this_thread::get_id() << std::endl;

	boost::thread th(boost::bind(update_prop, boost::ref(hive), boost::ref(o)));

	app.exec();

	return 0;
}
