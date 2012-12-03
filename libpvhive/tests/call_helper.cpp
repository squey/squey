#include <picviz/PVView.h>

#include <pvhive/PVHive.h>
#include <pvhive/PVActor.h>
#include <pvkernel/core/PVSharedPointer.h>
#include <pvhive/PVCallHelper.h>

#include <pvkernel/core/picviz_assert.h>

#define INT_CST1 42
#define INT_CST2 43
#define BOOL_CST true

struct Test
{
	void func(int i1, int i2)
	{
		PV_VALID(i1, INT_CST1);
		PV_VALID(i2, INT_CST2);
		std::cout << "func(int i1=" << i1 << ", int i2=" << i2 << ")" << std::endl;
	}
	int func2()
	{
		std::cout << "func2()" << std::endl;
		return INT_CST1;
	}

	void funcOv(int i1, int i2)
	{
		PV_VALID(i1, INT_CST1);
		PV_VALID(i2, INT_CST2);
		std::cout << "funcOv(int i1=" << i1 << ", int i2=" << i2 << ")" << std::endl;
	}
	void funcOv(bool b)
	{
		PV_VALID(b, BOOL_CST);
		std::cout << std::boolalpha << "funcOv(bool b=" << b << ")" << std::endl;
	}
};

int main()
{
	typedef PVCore::PVSharedPtr<Test> Test_p;
	Test_p test_p = Test_p(new Test());

	PVHive::PVActor<Test> actor;
	PVHive::PVHive::get().register_actor(test_p, actor);

	PVHive::call<FUNC(Test::func)>(actor, INT_CST1, INT_CST2);
	PVHive::call<FUNC(Test::func)>(test_p, INT_CST1, INT_CST2);

	PVHive::call<void (Test::*)(int, int), &Test::func>(test_p, INT_CST1, INT_CST2);
	PVHive::call<decltype(&Test::func), &Test::func>(test_p, INT_CST1, INT_CST2);

	PVHive::call<FUNC_PROTOTYPE(void, Test, funcOv, bool)>(test_p, BOOL_CST);
	PVHive::call<void(Test::*)(int, int), &Test::funcOv>(test_p, INT_CST1, INT_CST2);

	PV_VALID(PVHive::call<FUNC(Test::func2)>(test_p), INT_CST1);
	std::cout << PVHive::call<FUNC(Test::func2)>(test_p) << std::endl;
}
