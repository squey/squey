
#include <pvhive/PVHive.h>

int main()
{
	PVHive::PVHive &h1 = PVHive::PVHive::get();
	PVHive::PVHive &h2 = PVHive::PVHive::get();


	if (&h1 != &h2) {
		std::cerr << "PVHive::get() returns different addresses" << std::endl;
		return 1;
	}

	std::cout << "PVHive::get() works" << std::endl;

	return 0;
}
