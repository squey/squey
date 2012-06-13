#include <iostream>
#include <map>

#include <QApplication>

#include <boost/thread.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/bind.hpp>

class PVHiveTest
{
public:
	static PVHiveTest &get()
	{
		if (_hive == nullptr) {
			_hive = new PVHiveTest;
		}
		return *_hive;
	}

public:
	void register_actor()
	{
		{
			std::cout << "PVHiveTest::register_actor() trying to lock _actor_mutex (write)" << std::endl;
			boost::lock_guard<boost::mutex> lock(_actors_mutex);
			std::cout << "PVHiveTest::register_actor() _actor_mutex locked (write)" << std::endl;
			std::cout << "PVHiveTest::register_actor() begin (write) _actors" << std::endl;
			sleep(1); // insert actor
			std::cout << "PVHiveTest::register_actor() end (write) _actors" << std::endl;
		}
		std::cout << "PVHiveTest::register_actor() _actor_mutex released (write)" << std::endl;
		std::cout << "---" << std::endl;
	}

	void unregister_actor()
	{
		{
			std::cout << "PVHiveTest::unregister_actor() trying to lock _observers_lock (read)" << std::endl;
			read_lock_t read_lock(_observers_lock);
			std::cout << "PVHiveTest::unregister_actor() _observers_lock locked (read)" << std::endl;
			std::cout << "PVHiveTest::unregister_actor() begin (read) _observers" << std::endl;
			sleep(1); // read observers
			std::cout << "PVHiveTest::unregister_actor() end (read) _observers" << std::endl;
		}
		std::cout << "PVHiveTest::unregister_actor() _observers_lock released (read)" << std::endl;
		std::cout << "PVHiveTest::unregister_actor() trying to lock _actors_mutex (write)" << std::endl;
		{
			boost::lock_guard<boost::mutex> lock(_actors_mutex);
			std::cout << "PVHiveTest::unregister_actor() _actors_mutex locked (write)" << std::endl;
			std::cout << "PVHiveTest::unregister_actor() begin (write) _actors" << std::endl;
			sleep(1); // erase actor
			std::cout << "PVHiveTest::unregister_actor() end (write) _actors" << std::endl;
		}
		std::cout << "PVHiveTest::unregister_actor() _actors_mutex released (write)" << std::endl;

		std::cout << "---" << std::endl;
	}

	void register_observer()
	{
		{
			std::cout << "PVHiveTest::register_observer() trying to lock _observers_lock (write)" << std::endl;
			write_lock_t write_lock(_observers_lock);
			std::cout << "PVHiveTest::register_observer() _observers_lock locked (write)" << std::endl;

			std::cout << "PVHiveTest::register_observer() begin (write) _observers" << std::endl;
			sleep(1); // insert observer
			std::cout << "PVHiveTest::register_observer() end (write) _observers" << std::endl;
		}
		std::cout << "PVHiveTest::register_observer() _observers_lock released (write)" << std::endl;
		std::cout << "---" << std::endl;
	}

	void refresh_observers()
	{
		{
			std::cout << "PVHiveTest::refresh_observers() trying to lock _observers_lock (read)" << std::endl;
			read_lock_t read_lock(_observers_lock);
			std::cout << "PVHiveTest::refresh_observers() _observers_lock locked (read)" << std::endl;

			std::cout << "PVHiveTest::refresh_observers() begin (read) _observers" << std::endl;
			sleep(1); // insert observer
			std::cout << "PVHiveTest::refresh_observers() end (read) _observers" << std::endl;
		}
		std::cout << "PVHiveTest::refresh_observers() _observers_lock released (read)" << std::endl;
		std::cout << "---" << std::endl;
	}

private:
	static PVHiveTest *_hive;

	// thread safety
	typedef boost::shared_mutex lock_t;
	typedef boost::unique_lock<boost::shared_mutex> write_lock_t;
	typedef boost::shared_lock<boost::shared_mutex> read_lock_t;
	lock_t _observers_lock;
	boost::mutex _actors_mutex;
};

PVHiveTest *PVHiveTest::_hive = nullptr;

void thread1()
{
	PVHiveTest::get().register_actor();
	PVHiveTest::get().register_observer();
}

void thread2()
{
	sleep(1);
	PVHiveTest::get().refresh_observers();
}

void thread3()
{
	sleep(3);
	PVHiveTest::get().unregister_actor();
}

int main(int argc, char** argv)
{

	boost::thread th1(boost::bind(thread1));
	boost::thread th2(boost::bind(thread2));
	boost::thread th3(boost::bind(thread3));

	QApplication app(argc, argv);

	app.exec();

	return 0;
}
