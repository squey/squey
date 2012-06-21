
#include <iostream>

#include <QApplication>

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
};

class Obj1Observer : public PVHive::PVObserver<Obj1>
{
public:
	virtual void refresh()
	{
		std::cout << "  Obj1Observer::refresh for object " << _object << std::endl;
	}

	virtual void about_to_be_deleted()
	{
		std::cout << "    Obj1Observer::about_to_be_deleted for object " << _object << std::endl;
	}

private:
};

class Obj2Observer : public PVHive::PVObserver<Obj2>
{
public:
	virtual void refresh()
	{
		std::cout << "  Obj2Observer::refresh for object " << _object << std::endl;
	}

	virtual void about_to_be_deleted()
	{
		std::cout << "    Obj2Observer::about_to_be_deleted for object " << _object << std::endl;
	}

private:
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

	std::cout << "scene creation" << std::endl;

	std::cout << "  hive " << &h1 << std::endl;

	Obj1 *o1 = new Obj1;
	PVHive::PVActor<Obj1> a1o1;
	PVHive::PVHive::get().register_actor(*o1, a1o1);

	Obj2 *o2 = new Obj2;
	PVHive::PVActor<Obj2> a1o2;
	PVHive::PVHive::get().register_actor(*o2, a1o2);

	Obj1Observer o1o1;
	PVHive::PVHive::get().register_observer(*o1, o1o1);

	Obj2Observer o1o2;
	PVHive::PVHive::get().register_observer(*o2, o1o2);

	Obj2Observer o2o2;
	PVHive::PVHive::get().register_observer(*o2, o2o2);

	std::cout << "a1o1 calls &Obj1::print" << std::endl;
	PVACTOR_CALL(a1o1, &Obj1::print);
	std::cout << "a1o2 calls &Obj2::print" << std::endl;
	PVACTOR_CALL(a1o2, &Obj2::print);

	std::cout << "unregister a1o1" << std::endl;
	PVHive::PVHive::get().unregister_actor(a1o1);

	std::cout << "unregister o1" << std::endl;
	PVHive::PVHive::get().unregister_object(*o1);

	std::cout << "unregister o1o2" << std::endl;
	PVHive::PVHive::get().unregister_observer(o2o2);

	std::cout << "end" << std::endl;

	return 0;
}
