
#include <pvparallelview/PVZoomedZoneTree.h>

#include <iostream>

unsigned depth;
unsigned count;

void usage()
{
	std::cout << "usage: Tzoomedzonetree_impl depth count" << std::endl;
}

int main(int argc, char **argv)
{
	if (argc != 3) {
		usage();
		return 1;
	}

	depth = (unsigned)atoi(argv[1]);
	count = (unsigned)atoi(argv[2]);

	return 0;
}
