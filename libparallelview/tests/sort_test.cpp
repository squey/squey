
#include <iostream>
#include <vector>
#include <algorithm>

#include <tbb/parallel_sort.h>

#include <pvkernel/core/picviz_bench.h>

#include <pvparallelview/PVQuadTree.h>

#define MAX_VALUE ((1<<22) - 1)

unsigned count;

std::vector<PVParallelView::PVQuadTreeEntry> entries1;
std::vector<PVParallelView::PVQuadTreeEntry> entries2;

bool entry_less(const PVParallelView::PVQuadTreeEntry &a, const PVParallelView::PVQuadTreeEntry &b)
{
	if (a.y2 < b.y2)
		return true;
	else if (a.y2 == b.y2)
		return (a.y1 < b.y1);
	else
		return false;
}

void usage()
{
	std::cout << "usage: Tsort_test count" << std::endl;
}

int main(int argc, char **argv)
{
	if (argc != 2) {
		usage();
		return 1;
	}

	count = (unsigned)atoi(argv[1]);

	entries1.reserve(count);

	for(unsigned i = 0; i < count; ++i) {
		PVParallelView::PVQuadTreeEntry e;
		e.y1 = random() & MAX_VALUE;
		e.y2 = random() & MAX_VALUE;
		e.idx = i;
		entries1.push_back(e);
	}

	entries2 = entries1;

	{
		BENCH_START(sort);
		std::sort(entries1.begin(), entries1.end(), entry_less);
		BENCH_END(sort, "std::sort", 1, 1, 1, 1);
	}

	{
		BENCH_START(sort);
		tbb::parallel_sort(entries2.begin(), entries2.end(), entry_less);
		BENCH_END(sort, "tbb::sort", 1, 1, 1, 1);
	}

	return 0;
}
