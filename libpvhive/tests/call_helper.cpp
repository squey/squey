#include <picviz/PVView.h>

#include <pvhive/PVHive.h>
#include <pvhive/PVActor.h>
#include <pvkernel/core/PVSharedPointer.h>
#include <pvhive/PVCallHelper.h>

struct Test
{
	void func(int i1, int i2) { std::cout << "func(int i1, int i2)" << std::endl; }
	int func2() { std::cout << "func2()" << std::endl; return 42; }

	void funcOv(int i1, int i2) { std::cout << "funcOv(int i1, int i2)" << std::endl; }
	void funcOv(bool b) { std::cout << "funcOv(bool b)" << std::endl; }
};

int main()
{
	typedef PVCore::PVSharedPtr<Test> Test_p;
	Test_p test_p = Test_p(new Test());

	PVHive::PVActor<Test> actor;
	PVHive::PVHive::get().register_actor(test_p, actor);

	PVHive::call<FUNC(Test::func)>(actor, 42, 43);
	PVHive::call<FUNC(Test::func)>(test_p, 42, 43);

	PVHive::call<void (Test::*)(int, int), &Test::func>(test_p, 42, 43);
	PVHive::call<decltype(&Test::func), &Test::func>(test_p, 42, 43);

	PVHive::call<FUNC_PROTOTYPE(void, Test, funcOv, bool)>(test_p, true);
	PVHive::call<void(Test::*)(int, int), &Test::funcOv>(test_p, 42, 43);

	std::cout << PVHive::call<FUNC(Test::func2)>(test_p) << std::endl;
}
