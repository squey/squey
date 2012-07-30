#include <pvhive/PVHive.h>
#include <boost/shared_ptr.hpp>

class Test
{
public:
	Test() { std::cout << "Test::Test" << std::endl; }
	~Test() { std::cout << "~Test::Test" << std::endl; }
};

namespace PVHive_
{

template <typename T>
struct hive_deleter
{
	void operator()(T* ptr)
	{
		std::cout << "HIVE DELETER" << std::endl;
		//PVHive::get().unregister_object((void*) ptr);
		delete ptr;
	}
};

template <typename T>
boost::shared_ptr<T> make_shared(T* p)
{
	//return boost::shared_ptr<T>(p, hive_deleter(p));
}

}

int main()
{
	typedef boost::shared_ptr<Test> Test_p;
	Test_p test_p = PVHive_::make_shared(new Test());
}
