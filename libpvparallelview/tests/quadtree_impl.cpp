/**
 * \file quadtree_impl.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <iostream>
#include <vector>

#include <pvkernel/core/picviz_bench.h>

#include <pvparallelview/PVQuadTree.h>
#include <pvkernel/core/PVHSVColor.h>

unsigned count;
unsigned depth;

typedef PVParallelView::PVQuadTree<10000, 1000, 10000> pvquadtree;

pvquadtree *qt = 0;
PVParallelView::PVQuadTreeEntry *entries = 0;

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

	qt = new pvquadtree(0, MAX_VALUE, 0, MAX_VALUE, depth);

	std::cout << "Filling quadtree, it can take a while..." << std::endl;
	BENCH_START(fill);
	for(unsigned i = 0; i < count; ++i) {
		qt->insert(entries[i]);
	}
	BENCH_END(fill, "fill", 1, 1, 1, 1);

	std::cout << "sizeof(node): " << sizeof(*qt) << std::endl;
	std::cout << "memory used : " << qt->memory() << std::endl;

	PVParallelView::pv_quadtree_buffer_entry_t *buffer = new PVParallelView::pv_quadtree_buffer_entry_t [QUADTREE_BUFFER_SIZE];
	pvquadtree::pv_tlr_buffer_t *tlr = new pvquadtree::pv_tlr_buffer_t;

	for (unsigned i = 1; i < 9; ++i) {
		size_t num = 0;
		std::cout << "extract BCI codes from y1 for zoom " << i << std::endl;
		BENCH_START(extract);
		qt->get_first_from_y1(0, MAX_VALUE >> i, i, 1, buffer,
		                      [&](const PVParallelView::PVQuadTreeEntry &e,
		                          pvquadtree::pv_tlr_buffer_t &buffer)
		                      {
			                      (void)e;
			                      (void)buffer;
		                      }, *tlr);
		BENCH_END(extract, "extract", 1, 1, 1, 1);
		std::cout << "elements found: " << num << std::endl;
	}

	std::cout << std::endl;

	if (qt) {
		delete qt;
	}

	return 0;
}
