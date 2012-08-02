/**
 * \file multigrid_impl.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <iostream>
#include <vector>

#include <pvkernel/core/picviz_bench.h>

#include "PVMultiGrid.h"

unsigned count;
unsigned depth;

typedef PVParallelView::PVMultiGrid<2> multigrid_t;

multigrid_t *rmg;
PVParallelView::PVMultiGridEntry *entries;

#define MAX_VALUE ((1<<22) - 1)

void usage()
{
	std::cout << "usage: multigrid_impl depth count" << std::endl;
}

int main(int argc, char **argv)
{
	if (argc != 3) {
		usage();
		return 1;
	}

	depth = (unsigned)atoi(argv[1]);
	count = (unsigned)atoi(argv[2]);

	entries = new PVParallelView::PVMultiGridEntry [count];
	for(unsigned i = 0; i < count; ++i) {
		entries[i].y1 = random() & MAX_VALUE;
		entries[i].y2 = random() & MAX_VALUE;
		entries[i].idx = i;
	}

	rmg = new multigrid_t(0, MAX_VALUE, 0, MAX_VALUE, depth);

	std::cout << "Filling multigrid, it can take a while..." << std::endl;
	BENCH_START(fill);
	for(unsigned i = 0; i < count; ++i) {
		rmg->insert(entries[i]);
	}
	BENCH_END(fill, "fill", 1, 1, 1, 1);

	std::cout << "sizeof(node): " << sizeof(*rmg) << std::endl;
	std::cout << "memory used : " << rmg->memory() << std::endl;
	std::cout << "max_depth   : " << rmg->max_depth() << std::endl;

	//delete rmg;

	return 0;
}
