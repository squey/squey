
#include <iostream>
#include <vector>

#include <pvkernel/core/picviz_bench.h>

#include <pvparallelview/PVQuadTree.h>

unsigned count;
unsigned depth;

PVParallelView::PVQuadTree *qt;
PVParallelView::PVQuadTreeEntry *entries;

#define MAX_VALUE ((1<<22) - 1)

void usage()
{
	std::cout << "usage: test-quadtree depth count" << std::endl;
}

int main(int argc, char **argv)
{
	if (argc != 3) {
		usage();
		return 1;
	}

	depth = (unsigned)atoi(argv[1]);
	count = (unsigned)atoi(argv[2]);

	entries = new PVParallelView::PVQuadTreeEntry [count];
	for(unsigned i = 0; i < count; ++i) {
		entries[i].y1 = random() & MAX_VALUE;
		entries[i].y2 = random() & MAX_VALUE;
		entries[i].idx = i;
	}

	qt = new PVParallelView::PVQuadTree(0, MAX_VALUE, 0, MAX_VALUE, depth);

	std::cout << "Filling quadtree, it can take a while..." << std::endl;
	BENCH_START(fill_time);
	for(unsigned i = 0; i < count; ++i) {
		qt->insert(entries[i]);
	}
	BENCH_END(fill_time, "fill time", 1, 1, 1, 1);

	std::cout << "sizeof     : " << sizeof(*qt) << std::endl;
	std::cout << "memory used: " << qt->memory() << std::endl;

	return 0;
}
