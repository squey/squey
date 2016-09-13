/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/inendi_assert.h>

#include <pvparallelview/PVQuadTree.h>

#include "common.h"

#include <stdlib.h>

#define VALUE_MAX (1UL << 22)
#define VALUE_MASK (VALUE_MAX - 1)

typedef PVParallelView::PVQuadTree<> quadtree_t;
typedef PVParallelView::PVQuadTreeEntry quadtree_entry_t;

int main(int argc, char** argv)
{
	size_t num = 1000;
	if (argc >= 2) {
		num = atol(argv[1]);
	} else if (argc == 3) {
		num = atol(argv[1]);
		srand(atoi(argv[2]));
	} else {
		std::cout << "usage: " << basename(argv[0]) << ": num [seed]" << std::endl;
		return 0;
	}

	std::cout << "initialization, it can take a while" << std::endl;
	std::unique_ptr<quadtree_t> qt(new quadtree_t(0, VALUE_MAX, 0, VALUE_MAX, 8));
	for (size_t i = 0; i < num; ++i) {
		qt->insert(quadtree_entry_t(i, random() & VALUE_MASK, random() & VALUE_MASK));
	}
	std::cout << "done" << std::endl;

	return 0;
}
