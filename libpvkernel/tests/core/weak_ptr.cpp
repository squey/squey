#include <iostream>
#include <cassert>

#include <boost/shared_ptr.hpp>

#include <pvkernel/core/PVSharedPointer.h>
#include <pvkernel/core/PVWeakPointer.h>
#include <pvkernel/core/PVEnableSharedFromThis.h>

class Test1;
class Test2;
class Test3;

Test3* create_Test3();

typedef PVCore::PVSharedPtr<Test1> Test1_sp;
typedef PVCore::PVWeakPtr<Test1> Test1_wp;
typedef PVCore::PVSharedPtr<Test2> Test2_sp;
typedef PVCore::PVWeakPtr<Test2> Test2_wp;
typedef PVCore::PVSharedPtr<Test3> Test3_sp;
typedef PVCore::PVWeakPtr<Test3> Test3_wp;

static int g_deleter_count = 0;

template <typename T>
inline void deleter(T *ptr)
{
	std::cout << "deleter" << std::endl;
	g_deleter_count++;
	delete ptr;
}

struct Test1 : public PVCore::PVEnableSharedFromThis<Test1>
{
	Test1() { std::cout << "+Test1::Test1 (" << this << ")"<< std::endl; }
	virtual ~Test1() { std::cout << "~Test1::Test1 (" << this << ")"<< std::endl; }
};

struct Test2 : public Test1
{
	Test2() { std::cout << "+Test2::Test2 (" << this << ")"<< std::endl; }
	virtual ~Test2() { std::cout << "~Test2::Test2 (" << this << ")"<< std::endl; }
};

int main()
{
	Test1* test1_p1 = new Test1();
	Test1* test1_p2 = new Test1();
	Test1* test1_p3 = new Test1();
	Test1_sp test1_sp1(test1_p1);
	//test1_sp1.set_deleter(&deleter<Test1>);
	Test1_wp test1_wp1(test1_sp1);

	// weak pointer does not increment the counter itself
	std::cout << "test1_sp1.use_count()=" << test1_sp1.use_count() << std::endl;
	assert(test1_sp1.use_count() == 1);
	std::cout << "test1_wp1.use_count()=" << test1_wp1.use_count() << std::endl;
	assert(test1_wp1.use_count() == 1);

	// test1_p1->shared_from_this returns a PVSharedPtr with the proper pointer and count
	std::cout << "test1_p=" << test1_p1 << ", test1_p->shared_from_this().get()=" << test1_p1->shared_from_this().get() << std::endl;
	assert(test1_p1->shared_from_this().get() == test1_p1);
	std::cout << "test_p1->shared_from_this().use_count()=" << test1_p1->shared_from_this().use_count() << std::endl;
	assert(test1_p1->shared_from_this().use_count() == 2);

	// PVWeakPtr count is working properly (increment, decrement)
	{
		Test1_sp test1_sp2(test1_sp1);
		test1_sp2.set_deleter(&deleter<Test1>);
		Test1_sp test1_sp3(test1_sp2);
		test1_sp3.set_deleter(&deleter<Test1>);
		std::cout << "test1_wp1.use_count()=" << test1_wp1.use_count() << std::endl;
		assert(test1_wp1.use_count() == test1_sp2.use_count() && test1_sp2.use_count() == test1_sp3.use_count() && test1_sp3.use_count() == 3);
	}
	std::cout << "test1_wp1.use_count()=" << test1_wp1.use_count() << std::endl;
	assert(test1_wp1.use_count() == 1);

	// PVWeakPtr::lock() returns a PVSharedPtr with the proper pointer...
	std::cout << "test1_p=" << test1_p1 << std::endl;
	std::cout << "test1_wp1.lock().get()=" << test1_wp1.lock().get() << std::endl;
	assert(test1_wp1.lock().get() == test1_p1);

	// ...and the proper count
	std::cout << "test1_wp1.lock().use_count()=" << test1_wp1.lock().use_count() << std::endl;
	std::cout << "test1_sp1.use_count()=" << test1_sp1.use_count() << std::endl;
	assert(test1_wp1.lock().use_count() == test1_sp1.use_count());

	std::cout << "test1_wp1.expired()=" << std::boolalpha << test1_wp1.expired() << std::endl;
	assert(test1_wp1.expired() == false);

	std::cout << "test1_sp1.reset()" << std::endl;
	test1_sp1.reset();

	std::cout << "test1_wp1.use_count()=" << test1_wp1.use_count() << std::endl;
	assert(test1_wp1.use_count() == 0);

	std::cout << "test1_wp1.expired()=" << std::boolalpha << test1_wp1.expired() << std::endl;
	assert(test1_wp1.expired() == true);

	// Deleter is called one and exactly one time.
	std::cout << "g_deleter_count=" << g_deleter_count << std::endl;
	assert(g_deleter_count == 1);
	test1_sp1.reset(test1_p2);

	// test1_sp1.reset(test1_p2) correctly reset the pointer and the count
	std::cout << "test1_p2=" << test1_p2 << ", test1_sp1.get()=" << test1_sp1.get() << std::endl;
	assert(test1_sp1.get() == test1_p2);
	std::cout << "test1_sp1.use_count()=" << test1_sp1.use_count() << std::endl;
	assert(test1_sp1.use_count() == 1);
	test1_sp1.reset();

	// test1_sp1.reset(test1_p3, &deleter<Test1>) correctly reset the pointer, the deleter and the count
	test1_sp1.reset(test1_p3, &deleter<Test1>);
	std::cout << "test1_wp1.use_count()=" << test1_wp1.use_count() << std::endl;
	std::cout << "test1_p3=" << test1_p3 << ", test1_sp1.get()=" << test1_sp1.get() << std::endl;
	assert(test1_sp1.get() == test1_p3);
	std::cout << "test1_sp1.use_count()=" << test1_sp1.use_count() << std::endl;
	assert(test1_sp1.use_count() == 1);
	g_deleter_count = 0;
	test1_sp1.reset();
	std::cout << "g_deleter_count=" << g_deleter_count << std::endl;
	assert(g_deleter_count == 1);

	Test2* test2_p1 = new Test2();
	Test2_sp test2_sp1(test2_p1);

	// (Note: Test2 is not directly derived from PVSharedFromThis) test2_p1->shared_from_this
	// returns a PVSharedPtr with the proper pointer and the proper count
	std::cout << "test2_p1=" << test2_p1 << ", test2_p1->shared_from_this().get()=" << test2_p1->shared_from_this().get() << std::endl;
	assert(test2_p1->shared_from_this().get() == test2_p1);
	std::cout << "test2_p1->shared_from_this().use_count()=" << test2_p1->shared_from_this().use_count() << std::endl;
	assert(test2_p1->shared_from_this().use_count() == 2);

	// PVWeakPtr count is 0 after a PVSharedPtr::reset(*)
	std::cout << "test1_wp1.use_count()=" << test1_wp1.use_count() << std::endl;
	assert(test1_wp1.use_count() == 0);

	// Test incomplete type
	Test3* test3_p1 = create_Test3();
	Test3_sp test3_sp1(test3_p1, &deleter<Test3>);
	std::cout << "test3_sp1.use_count()=" << test3_sp1.use_count() << std::endl;
	assert(test3_sp1.use_count() == 1);
	g_deleter_count = 0;
	test3_sp1.reset();
	std::cout << "g_deleter_count=" << g_deleter_count << std::endl;
	assert(g_deleter_count == 1);

	// Test PVSharedPtr creation from a PVShared pointer of a derived type
	Test2* test2_p2 = new Test2();
	Test2* test2_p3 = new Test2();
	Test2_sp test2_sp2 = Test2_sp(test2_p2);
	Test2_sp test2_sp3(test2_p3);
	Test1_sp test1_sp4(test2_sp2);
	std::cout << "test1_sp4.get()=" << test1_sp4.get() << ", test2_sp2.get()=" << test2_sp2.get() << ", test2_sp3.get()=" << test2_sp3.get() << std::endl;
	assert(test1_sp4.get() == test2_sp2.get());

	// Test PVSharedPtr creation from a pointer of a derived type
	Test2* test2_p4 = new Test2();
	Test1_sp test1_sp5(test2_p4); // Warning: unlikely to boost::shared_ptr, the derived destructor has to be virtual to be called.

	std::cout << "test1_sp5.get()=" << test1_sp5.get() << ", test2_p4)=" << test2_p4 << std::endl;
	assert(test1_sp5.get() == test2_p4);

	// Tests from non derived types (invalid, will not compile)
	//Test1_sp test1_sp6(create_Test3());
	//Test3* test3_p2 = create_Test3();
	//Test3_sp test3_sp2(test3_p2);
	//Test1_sp test1_sp7(test3_sp2);

	return 0;
}

struct Test3
{
	Test3() { std::cout << "+Test3::Test3 (" << this << ")"<< std::endl; }
	virtual ~Test3() { std::cout << "~Test3::Test3 (" << this << ")"<< std::endl; }
};

Test3* create_Test3() { return new Test3(); }
