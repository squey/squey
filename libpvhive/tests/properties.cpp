
#include <pvhive/PVHive.h>
#include <pvhive/PVActor.h>
#include <pvhive/PVObserver.h>

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

class MyObjectObserver : public PVHive::PVObserver<MyObjectProperty>
{
public:
	MyObjectObserver() {}

	void refresh()
	{
		std::cout << "  MyObjectObserver::refresh for object "
		          << _object << std::endl;
	}

	void about_to_be_deleted()
	{
		std::cout << "  MyObjectObserver::about_to_be_deleted for object "
		          << _object << std::endl;
	}
};


class MyObjectPropertyObserver : public PVHive::PVObserver<MyObjectProperty>
{
public:
	MyObjectPropertyObserver() {}

	void refresh()
	{
		std::cout << "  MyObjectPropertyObserver::refresh for object "
		          << _object << std::endl;
	}

	void about_to_be_deleted()
	{
		std::cout << "  MyObjectPropertyObserver::about_to_be_deleted for object "
		          << _object << std::endl;
	}
};

int main()
{
	PVHive::PVHive &hive = PVHive::PVHive::get();

	MyObject myobj(42);
	hive.register_object(myobj);
	hive.register_object(myobj, [](MyObject const &myo) -> const MyObjectProperty &
	                     {
		                     return myo.get_prop();
	                     });


	// 1 acteur sur myobj
	PVHive::PVActor<MyObject> oactor;

	hive.register_actor(myobj, oactor);

	// 1 acteur sur myobj.prop
	PVHive::PVActor<MyObjectProperty> pactor;

	hive.register_actor(myobj.get_prop(), pactor);

	// 1 observeur sur myobj
	MyObjectObserver oobserver;
	std::cout << "register observer " << &oobserver
	          << " for myobj " << &myobj << std::endl;
	hive.register_observer(myobj, oobserver);

	// 2 observeur sur myobj.prop, 1 en lambda et un en methode
	MyObjectPropertyObserver pobserver;
	std::cout << "register observer " << &pobserver
	          << " for myobj.get_prop() " << &(myobj.get_prop()) << std::endl;
	hive.register_observer(myobj, [](MyObject const &myo) -> const MyObjectProperty &
	                       {
		                       return myo.get_prop();
	                       }, pobserver);

	std::cout << "oactor.call(8)" << std::endl;
	oactor.call<decltype(&MyObject::set_i), &MyObject::set_i>(8);

	std::cout << "pactor.call(42)" << std::endl;
	pactor.call<decltype(&MyObjectProperty::set_value), &MyObjectProperty::set_value>(42);

	std::cout << "hive.unregister_actor(oactor)" << std::endl;
	hive.unregister_actor(oactor);

	std::cout << "hive.unregister_actor(pactor)" << std::endl;
	hive.unregister_actor(pactor);

	std::cout << "hive.unregister_object(myobj)" << std::endl;
	hive.unregister_object(myobj);

	return 0;
}






