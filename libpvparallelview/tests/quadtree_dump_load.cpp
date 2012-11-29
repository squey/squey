
#include <pvkernel/core/picviz_assert.h>

#include <pvparallelview/PVQuadTree.h>

#include "common.h"

#include <stdlib.h>

#define VALUE_MAX (1UL<<22)
#define VALUE_MASK (VALUE_MAX - 1)

#define FILENAME "quadtree.dump"

typedef PVParallelView::PVQuadTree<> quadtree_t;
typedef PVParallelView::PVQuadTreeEntry quadtree_entry_t;

void clean()
{
	remove(FILENAME);
}

int main(int argc, char **argv)
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

	atexit(clean);

#ifdef PICVIZ_DEVELOPER_MODE
	quadtree_t *qt;
	quadtree_t* qt2;

	std::cout << "initialization, it can take a while" << std::endl;
	qt = new quadtree_t(0, VALUE_MAX, 0, VALUE_MAX, 8);
	for (size_t i = 0; i < num; ++i) {
		qt->insert(quadtree_entry_t(i, random() & VALUE_MASK, random() & VALUE_MASK));
	}
	std::cout << "done" << std::endl;

	std::cout << "dumping" << std::endl;
	bool ret = qt->dump_to_file(FILENAME);
	PV_VALID(ret, true);
	std::cout << "done" << std::endl;

	std::cout << "exhuming" << std::endl;
	qt2 = quadtree_t::load_from_file(FILENAME);
	PV_ASSERT_VALID(qt2 != nullptr);
	std::cout << "done" << std::endl;

	ret = (*qt == *qt2);
	PV_VALID(ret, true);
#endif

	return 0;
}
