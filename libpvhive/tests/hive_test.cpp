
#include <iostream>

#include <pvhive/PVHive.h>
#include <pvhive/PVActor.h>

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

	return 0;
}
