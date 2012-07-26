/**
 * \file multithreads.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <boost/thread.hpp>

#include <pvhive/PVHive.h>
#include <pvhive/PVActor.h>
#include <pvhive/PVObserver.h>

#include <pvkernel/core/PVLogger.h>
#include <pvkernel/core/PVSharedPointer.h>

#define MAX_COUNT 100000

struct Obj1
{
	void test()
	{
	}
};

typedef PVCore::PVSharedptr<Obj1> Obj1_p;

class Obj1Observer : public PVHive::PVObserver<Obj1>
{
public:
	virtual void refresh()
	{
	}

	virtual void about_to_be_deleted()
	{
	}

private:
};

void thread1(Obj1_p& obj, PVHive::PVActor<Obj1>& actor, uint32_t &nb_success, uint32_t &nb_exception)
{
	PVHive::PVHive::get().register_actor(obj, actor);
	for(int i = 0; i < MAX_COUNT; i++) {
		try {
			boost::this_thread::sleep(boost::posix_time::microseconds(rand()%2));
			PVACTOR_CALL(actor, &Obj1::test);
			nb_success++;
		}
		catch(PVHive::no_object) {
			PVHive::PVHive::get().register_actor(obj, actor);
			nb_exception++;
		}
	}
}

void thread2(PVHive::PVActor<Obj1>& actor, uint32_t &count)
{
	for(int i = 0 ; i < MAX_COUNT; i++) {
		count += PVHive::PVHive::get().unregister_actor(actor);
		boost::this_thread::sleep(boost::posix_time::microseconds(rand()%2));
	}
}

int main()
{
	srand(time(NULL));

	bool test1_monothread_no_object_exception_raised = false;
	bool test2_multithread_no_object_exception_raised = false;

	///////////////////////////////////////////////////
	//  Test1 - Monothreaded no_object exception
	///////////////////////////////////////////////////
	{
	PVHive::PVActor<Obj1> a1o1;
	Obj1Observer o1o1;

	{
	Obj1_p o1 = Obj1_p(new Obj1);
	PVHive::PVHive::get().register_actor(o1, a1o1);
	PVHive::PVHive::get().register_observer(o1, o1o1);
	}

	try {
		PVACTOR_CALL(a1o1, &Obj1::test);
	}
	catch(PVHive::no_object) {
		test1_monothread_no_object_exception_raised = true;
	}

	PVLOG_INFO("no_object monothreaded exception passed: %d\n", test1_monothread_no_object_exception_raised);
	}

	///////////////////////////////////////////////////
	//  Test2 - Multithreaded no_object exception
	///////////////////////////////////////////////////
	{
	Obj1_p obj = Obj1_p(new Obj1);
	PVHive::PVActor<Obj1> actor;

	uint32_t nb_sucess = 0;
	uint32_t nb_exceptions = 0;
	uint32_t nb_unregistered = 0;
	boost::thread th1(boost::bind(thread1, boost::ref(obj), boost::ref(actor), boost::ref(nb_sucess), boost::ref(nb_exceptions)));
	boost::thread th2(boost::bind(thread2, boost::ref(actor), boost::ref(nb_unregistered)));

	th1.join();
	th2.join();

	test2_multithread_no_object_exception_raised = (nb_exceptions == nb_unregistered || nb_exceptions == nb_unregistered-1) && (nb_sucess + nb_exceptions == MAX_COUNT) ;

	PVLOG_INFO("no_object multithreaded exception passed: %d (success:%d / exceptions:%d / total:%d, unregister:%d)\n", test2_multithread_no_object_exception_raised, nb_sucess, nb_exceptions, MAX_COUNT, nb_unregistered);
	}

	return !(test1_monothread_no_object_exception_raised && test2_multithread_no_object_exception_raised);
}
