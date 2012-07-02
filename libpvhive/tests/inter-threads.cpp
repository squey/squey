
#include <boost/thread.hpp>

#include <pvhive/PVHive.h>
#include <pvhive/PVActor.h>

class Entity
{
public:
	Entity(int i) : _i(i)
	{}

	void set_i(int i)
	{
		_i = i;
	}

	int get_i() const
	{
		return _i;
	}

private:
	int _i;
};

Entity *e = nullptr;

static bool observer_must_run = true;

typedef PVHive::PVActor<Entity> EntityActor;

class EntityObserver : public PVHive::PVObserver<Entity>
{
public:
	EntityObserver()
	{
	}

	void set_entity(Entity *e)
	{
		_e = e;
	}

	void refresh()
	{
		std::cout << "  ::refresh thread("
		          << boost::this_thread::get_id()
		          << ") i(" << e->get_i() << ")" << std::endl;
	}

	void about_to_be_deleted()
	{
		observer_must_run = false;
		std::cout << "  ::about_to_be_deleted thread("
		          << boost::this_thread::get_id() << ")"
		          << std::endl;
	}

private:
	Entity *_e;
};

void th_actor_func()
{
	std::cout << "th_actor: init - " <<boost::this_thread::get_id()
	          << std::endl;
	int count = 0;
	e = new Entity(42);
	EntityActor a;
	PVHive::PVHive::get().register_actor(*e, a);

	std::cout << "th_actor: pseudo sync" << std::endl;
	sleep(1);

	std::cout << "th_actor: run" << std::endl;
	while (count < 10) {
		sleep(1);
		std::cout << "th_actor_func - " << boost::this_thread::get_id()
		          << " - e.set_i(" << count << ")" << std::endl;
		PVACTOR_CALL(a, &Entity::set_i, count);
		++count;
	}

	std::cout << "th_actor: clean" << std::endl;
	PVHive::PVHive::get().unregister_actor(a);
	PVHive::PVHive::get().unregister_object(*e);
	delete e;
	e = nullptr;

	std::cout << "th_actor: terminate" << std::endl;
}

/* This thread will wait until 'e' is destroyed by its thread.
 */
void th_long_observer_func()
{
	std::cout << "th_long_observer_func: init - "
	          << boost::this_thread::get_id()  << std::endl;
	EntityObserver o;
	PVHive::PVHive::get().register_observer(*e, o);

	std::cout << "th_long_observer_func: run" << std::endl;
	while (observer_must_run) {
		sleep(1);
	}

	std::cout << "th_long_observer_func: clean" << std::endl;
	PVHive::PVHive::get().unregister_observer(o);

	std::cout << "th_long_observer_func: terminate" << std::endl;
}

/* This thread will wait less time to exit earlier.
 */
void th_short_observer_func()
{
	std::cout << "th_short_observer_func: init - "
	          << boost::this_thread::get_id()  << std::endl;

	int count = 0;
	EntityObserver o;

	PVHive::PVHive::get().register_observer(*e, o);

	std::cout << "th_short_observer_func: run" << std::endl;

	while (count < 5) {
		sleep(1);
		++count;
	}

	std::cout << "th_short_observer_func: clean" << std::endl;
	PVHive::PVHive::get().unregister_observer(o);

	std::cout << "th_short_observer_func: terminate" << std::endl;
}

int main()
{

	boost::thread tha(boost::bind(th_actor_func));
	sleep(1);
	boost::thread thlo(boost::bind(th_long_observer_func));
	boost::thread thso(boost::bind(th_short_observer_func));

	tha.join();
	thlo.join();
	thso.join();

	PVHive::PVHive::get().print();

	return 0;
}
