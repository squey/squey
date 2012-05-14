
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
	BENCH_START(fill);
	for(unsigned i = 0; i < count; ++i) {
		qt->insert(entries[i]);
	}
	BENCH_END(fill, "fill", 1, 1, 1, 1);

	std::cout << "sizeof(node): " << sizeof(*qt) << std::endl;
	std::cout << "memory used : " << qt->memory() << std::endl;


	PVParallelView::PVQuadTree *subtree = 0;

	{
		std::cout << "extract from full y1" << std::endl;
		BENCH_START(extract);
		subtree = qt->get_subtree_from_y1(0, MAX_VALUE);
		BENCH_END(extract, "extract", 1, 1, 1, 1);
		std::cout << "memory used: " << subtree->memory() << std::endl;
		delete subtree;
	}

	{
		std::cout << "extract from half y1" << std::endl;
		BENCH_START(extract);
		subtree = qt->get_subtree_from_y1(0, MAX_VALUE >> 1);
		BENCH_END(extract, "extract", 1, 1, 1, 1);
		std::cout << "memory used: " << subtree->memory() << std::endl;
		delete subtree;
	}

	{
		std::cout << "extract from full y1y2" << std::endl;
		BENCH_START(extract);
		subtree = qt->get_subtree_from_y1y2(0, MAX_VALUE, 0, MAX_VALUE);
		BENCH_END(extract, "extract", 1, 1, 1, 1);
		std::cout << "memory used: " << subtree->memory() << std::endl;
		delete subtree;
	}

	{
		std::cout << "extract from quarter y1y2" << std::endl;
		BENCH_START(extract);
		subtree = qt->get_subtree_from_y1y2(0, MAX_VALUE >> 1, 0, MAX_VALUE >> 1);
		BENCH_END(extract, "extract", 1, 1, 1, 1);
		std::cout << "memory used: " << subtree->memory() << std::endl;
		delete subtree;
	}

	Picviz::PVSelection *selection;
	selection = new Picviz::PVSelection();

	{
		std::cout << "extract from full selection" << std::endl;
		selection->select_all();
		BENCH_START(extract);
		subtree = qt->get_subtree_from_selection(*selection);
		BENCH_END(extract, "extract", 1, 1, 1, 1);
		std::cout << "memory used: " << subtree->memory() << std::endl;
		delete subtree;
	}

	{
		std::cout << "extract from half of selection" << std::endl;
		selection->select_even();
		BENCH_START(extract);
		subtree = qt->get_subtree_from_selection(*selection);
		BENCH_END(extract, "extract", 1, 1, 1, 1);
		std::cout << "memory used: " << subtree->memory() << std::endl;
		delete subtree;
	}

	{
		std::cout << "extract from quarter of selection" << std::endl;
		memset(selection->get_buffer(), 0x88, PICVIZ_SELECTION_NUMBER_OF_BYTES);
		BENCH_START(extract);
		subtree = qt->get_subtree_from_selection(*selection);
		BENCH_END(extract, "extract", 1, 1, 1, 1);
		std::cout << "memory used: " << subtree->memory() << std::endl;
		delete subtree;
	}

	{
		std::cout << "extract from no selection" << std::endl;
		selection->select_none();
		BENCH_START(extract);
		subtree = qt->get_subtree_from_selection(*selection);
		BENCH_END(extract, "extract", 1, 1, 1, 1);
		std::cout << "memory used: " << subtree->memory() << std::endl;
		delete subtree;
	}

	delete qt;

	return 0;
}
