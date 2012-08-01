#include <iostream>

#include <pvkernel/core/PVSharedPointer.h>
#include <pvkernel/core/PVEnableSharedFromThis.h>
#include <pvkernel/core/PVWeakPointer.h>

class Test;
class Test2;

typedef PVCore::PVSharedPtr<Test> Test_sp;
typedef PVCore::PVSharedPtr<Test2> Test2_sp;
typedef PVCore::PVWeakPtr<Test> Test_wp;

template <typename T>
inline void deleter(T *ptr)
{
	std::cout << "deleter" << std::endl;
	delete ptr;
}

class Test : public PVCore::PVEnableSharedFromThis<Test>
{
public:
	Test() { std::cout << "Test::Test" << std::endl; }
	~Test() { std::cout << "~Test::Test" << std::endl; }
	void print() { std::cout << "print" << std::endl; }
	void sft()
	{
		Test_sp test_sp = shared_from_this();
		std::cout << "shared_from_this().use_count():" << test_sp.use_count() << std::endl;
	}
};

struct Test2 : public Test
{
	void test() { std::cout << "Test2::shared_from_this().use_count()=" << shared_from_this().use_count() << std::endl; }
};

int main()
{
	Test_sp test_sp1(new Test());

	std::cout << "test_sp1.use_count()=" << test_sp1.use_count() << std::endl;

	test_sp1->sft();


	Test2_sp test2_sp(new Test2());
	test2_sp->test();

	std::cout << "test_sp1.use_count()=" << test_sp1.use_count() << std::endl;

	test_sp1.set_deleter(&deleter<Test>);
	Test_sp test_sp2(test_sp1);

	std::cout << "test_sp1.use_count()=" << test_sp1.use_count() << std::endl;

	Test_wp test_wp(test_sp1);
	Test_wp test_wp2(test_wp);

	std::cout << "test_wp.use_count()=" << test_wp.use_count() << std::endl;
	std::cout << "test_wp.expired()=" << test_wp.expired() << std::endl;

	test_wp.lock()->print();

	test_wp.lock()->sft();


	std::cout << "about to test_sp.reset();" << std::endl;
	test_sp1.reset();
	test_sp2.reset();

	std::cout << "test_wp.use_count()=" << test_wp.use_count() << std::endl;
	std::cout << "test_wp.expired()=" << test_wp.expired() << std::endl;
}
