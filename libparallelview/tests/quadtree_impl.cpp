
#include <iostream>
#include <vector>

#include <pvkernel/core/picviz_bench.h>

#include <pvparallelview/PVQuadTree.h>
#include <pvparallelview/PVHSVColor.h>

unsigned count;
unsigned depth;

PVParallelView::PVQuadTree *qt = 0;
PVParallelView::PVQuadTreeEntry *entries = 0;
PVParallelView::PVBCICode* bci_codes = 0;

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

	PVParallelView::PVHSVColor* colors = PVParallelView::PVHSVColor::init_colors(count);

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

	bci_codes = PVParallelView::PVBCICode::allocate_codes(4096);

	for (unsigned i = 1; i < 9; ++i) {
		std::cout << "extract BCI codes from y1 for zoom " << i << std::endl;
		BENCH_START(extract);
		size_t num = qt->get_first_bci_from_y1(0, MAX_VALUE >> i, i, colors, bci_codes);
		BENCH_END(extract, "extract", 1, 1, 1, 1);
		std::cout << "elements found: " << num << std::endl;
	}

	std::cout << std::endl;
	PVParallelView::PVQuadTree *subtree = 0;

	{
		std::cout << "extract subtree from full y1" << std::endl;
		BENCH_START(extract);
		subtree = qt->get_subtree_from_y1(0, MAX_VALUE);
		BENCH_END(extract, "extract", 1, 1, 1, 1);
		std::cout << "memory used: " << subtree->memory() << std::endl;
		delete subtree;
	}

	{
		std::cout << "extract subtree from half y1" << std::endl;
		BENCH_START(extract);
		subtree = qt->get_subtree_from_y1(0, MAX_VALUE >> 1);
		BENCH_END(extract, "extract", 1, 1, 1, 1);
		std::cout << "memory used: " << subtree->memory() << std::endl;
		delete subtree;
	}

	{
		std::cout << "extract subtree from full y1y2" << std::endl;
		BENCH_START(extract);
		subtree = qt->get_subtree_from_y1y2(0, MAX_VALUE, 0, MAX_VALUE);
		BENCH_END(extract, "extract", 1, 1, 1, 1);
		std::cout << "memory used: " << subtree->memory() << std::endl;
		delete subtree;
	}

	{
		std::cout << "extract subtree from quarter y1y2" << std::endl;
		BENCH_START(extract);
		subtree = qt->get_subtree_from_y1y2(0, MAX_VALUE >> 1, 0, MAX_VALUE >> 1);
		BENCH_END(extract, "extract", 1, 1, 1, 1);
		std::cout << "memory used: " << subtree->memory() << std::endl;
		delete subtree;
	}

	std::cout << std::endl;
	Picviz::PVSelection *selection;
	selection = new Picviz::PVSelection();

	{
		std::cout << "extract subtree from full selection" << std::endl;
		selection->select_all();
		BENCH_START(extract);
		subtree = qt->get_subtree_from_selection(*selection);
		BENCH_END(extract, "extract", 1, 1, 1, 1);
		std::cout << "memory used: " << subtree->memory() << std::endl;
		delete subtree;
	}

	{
		std::cout << "extract subtree from half of selection" << std::endl;
		selection->select_even();
		BENCH_START(extract);
		subtree = qt->get_subtree_from_selection(*selection);
		BENCH_END(extract, "extract", 1, 1, 1, 1);
		std::cout << "memory used: " << subtree->memory() << std::endl;
		delete subtree;
	}

	{
		std::cout << "extract subtree from quarter of selection" << std::endl;
		memset(selection->get_buffer(), 0x88, PICVIZ_SELECTION_NUMBER_OF_BYTES);
		BENCH_START(extract);
		subtree = qt->get_subtree_from_selection(*selection);
		BENCH_END(extract, "extract", 1, 1, 1, 1);
		std::cout << "memory used: " << subtree->memory() << std::endl;
		delete subtree;
	}

	{
		std::cout << "extract subtree from no selection" << std::endl;
		selection->select_none();
		BENCH_START(extract);
		subtree = qt->get_subtree_from_selection(*selection);
		BENCH_END(extract, "extract", 1, 1, 1, 1);
		std::cout << "memory used: " << subtree->memory() << std::endl;
		delete subtree;
	}

	if (qt) {
		delete qt;
	}
	// a double free occurs if uncommented...
	// if (bci_codes) { delete bci_codes; }

	return 0;
}
