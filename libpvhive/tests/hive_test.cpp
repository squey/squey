
#include <iostream>

#include <pvhive/PVHive.h>
#include <pvhive/PVActor.h>
#include <pvhive/PVObserver.h>

template <class T>
class Obs : public PVHive::PVObserver<T>
{
public:
	Obs()
	{}

	virtual void refresh()
	{
		std::cout << "Obs::refresh()" << std::endl;
	}

	virtual void about_to_be_deleted()
	{
		std::cout << "Obs::about_to_be_deleted()" << std::endl;
	}
};

int main()
{
	PVHive::PVHive &h1 = PVHive::PVHive::get();
	PVHive::PVHive &h2 = PVHive::PVHive::get();


	if (&h1 != &h2) {
		std::cerr << "PVHive::get() returns different addresses" << std::endl;
		return 1;
	}

	std::cout << "PVHive::get() works" << std::endl;

	int i;
	PVHive::PVActor<int> a;

	PVHive::PVHive::get().register_actor<int>(i, a);

	Obs<int> o;

	PVHive::PVHive::get().register_observer<int>(i, o);


	return 0;
}
