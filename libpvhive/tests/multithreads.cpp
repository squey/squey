#include <boost/thread.hpp>

#include <pvhive/PVHive.h>
#include <pvhive/PVActor.h>
#include <pvhive/PVObserver.h>

#include <pvkernel/core/PVLogger.h>
#include <pvkernel/core/PVSharedPointer.h>


struct Obj1
{
	void test()
	{
	}
};

typedef PVCore::pv_shared_ptr<Obj1> Obj1_p;

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

void thread1(Obj1_p& obj, PVHive::PVActor<Obj1>& actor, boost::promise<bool>& res)
{
	PVHive::PVHive::get().register_actor(obj, actor);
	while(true) {
		try {
			PVACTOR_CALL(actor, &Obj1::test);
		}
		catch(PVHive::no_object) {
			res.set_value(true);
			break;
		}
	}
}

bool thread2(Obj1_p& obj, PVHive::PVActor<Obj1>& actor, boost::promise<bool>& res)
{
	sleep(1);
	PVHive::PVHive::get().unregister_actor(actor);
	sleep(1);
	res.set_value(false);
}

int main(int argc, char** argv)
{
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

	boost::promise<bool> res;
	boost::thread th1(boost::bind(thread1, boost::ref(obj), boost::ref(actor), boost::ref(res)));
	boost::thread th2(boost::bind(thread2, boost::ref(obj), boost::ref(actor), boost::ref(res)));

	test2_multithread_no_object_exception_raised = res.get_future().get();

	PVLOG_INFO("no_object multithreaded exception passed: %d\n", test2_multithread_no_object_exception_raised);
	}

	return !(test1_monothread_no_object_exception_raised && test2_multithread_no_object_exception_raised);
}
