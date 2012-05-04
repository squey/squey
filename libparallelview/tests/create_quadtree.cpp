
#include <iostream>
#include <vector>
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>

#include <boost/random.hpp>
#include <boost/random/uniform_int_distribution.hpp>

#include <boost/math/distributions/normal.hpp>

#include <pvkernel/core/picviz_bench.h>

// gros hack pour que les quadtree connaissent la structure entry

#pragma pack(push)
#pragma pack(4)

struct entry {
	uint32_t y1, y2;
	uint32_t idx;
};

bool are_diff(const entry &e1, const entry &e2)
{
	return ((e1.y1 != e2.y1) || (e1.y2 != e2.y2) || (e1.idx != e2.idx));
}

#pragma pack(pop)

enum {
	SW = 0,
	SE,
	NW,
	NE
};

#include "quadtree.h"
#include "quadtree-tmpl.h"
#include "quadtree-flat.h"

void printb (uint32_t v)
{
	for(int i = 31; i >= 0; --i) {
		if((i & 7) == 7)
			std::cout << " ";
		if(v & (1 << i))
			std::cout << "1";
		else
			std::cout << "0";
	}
}

#define COUNT 100000000

#define MAX_VALUE ((1<<22) - 1)

void usage()
{
	std::cout << "usage: test-quadtree tree-level test-num" << std::endl;
	std::cout << std::endl;
	std::cout << "test 0: 	PVQuadTree with Vector1" << std::endl;
	std::cout << "test 1: 	PVQuadTree with Vector2" << std::endl;
	std::cout << "test 2: 	PVQuadTreeTmpl with Vector1" << std::endl;
	std::cout << "test 3: 	PVQuadTreeTmpl with Vector2" << std::endl;
	// std::cout << "test 4: 	PVQuadTreeFlat with Vector1" << std::endl;
	// std::cout << "test 5: 	PVQuadTreeFlat with Vector2" << std::endl;
}

int main(int argc, char **argv)
{
	if (argc != 3) {
		usage();
		return 1;
	}

	int depth = atoi(argv[1]);
	int test = atoi(argv[2]);

	if(test > 3) {
		usage();
		return 2;
	}

	boost::mt19937 rnd(0);
	boost::random::uniform_int_distribution<unsigned> uni(0, UINT_MAX);

	entry *entries = new entry  [COUNT];
	for(int i = 0; i < COUNT; ++i) {
		entries[i].y1 = random() & MAX_VALUE;
		entries[i].y2 = random() & MAX_VALUE;
		entries[i].idx = i;
	}

	if (test == 0) {
		PVQuadTree<Vector1<entry> > sqt1(0, MAX_VALUE, 0, MAX_VALUE, depth);
		std::cout << "sizeof(sqt1): " << sizeof(sqt1) << std::endl;
		MEM_START(usage);
		BENCH_START(time);
		for(int i = 0; i < COUNT; ++i) {
			sqt1.insert(entries[i]);
		}
		BENCH_END(time, "PVQuadTree Vector1", COUNT, sizeof(entry), 1, 1);
		MEM_END(usage, "PVQuadTree Vector1");
	}

	if (test == 1) {
		PVQuadTree<Vector2<entry> > sqt2(0, MAX_VALUE, 0, MAX_VALUE, depth);
		std::cout << "sizeof(sqt2): " << sizeof(sqt2) << std::endl;
		MEM_START(usage);
		BENCH_START(time);
		for(int i = 0; i < COUNT; ++i) {
			sqt2.insert(entries[i]);
		}
		BENCH_END(time, "PVQuadTree Vector2", COUNT, sizeof(entry), 1, 1);
		MEM_END(usage, "PVQuadTree Vector2");
	}

	if (test == 2) {
		PVQuadTreeTmpl<Vector1<entry>,8> *tqt1 = new PVQuadTreeTmpl<Vector1<entry>,8>(0, MAX_VALUE, 0, MAX_VALUE, 8);
		(void) depth;
		std::cout << "sizeof(tqt1): " << sizeof(*tqt1) << std::endl;
		MEM_START(usage);
		BENCH_START(time);
		for(int i = 0; i < COUNT; ++i) {
			tqt1->insert(entries[i]);
		}
		BENCH_END(time, "PVQuadTreeTmpl Vector1", COUNT, sizeof(entry), 1, 1);
		MEM_END(usage, "PVQuadTreeTmpl Vector1");
		delete tqt1;
	}

	if (test == 3)  {
		PVQuadTreeTmpl<Vector2<entry>,8> *tqt2 = new PVQuadTreeTmpl<Vector2<entry>,8>(0, MAX_VALUE, 0, MAX_VALUE, 8);
		(void) depth;
		std::cout << "sizeof(tqt2): " << sizeof(*tqt2) << std::endl;
		MEM_START(usage);
		BENCH_START(time);
		for(int i = 0; i < COUNT; ++i) {
			tqt2->insert(entries[i]);
		}
		BENCH_END(time, "PVQuadTreeTmpl Vector2", COUNT, sizeof(entry), 1, 1);
		MEM_END(usage, "PVQuadTreeTmpl Vector2");
		delete tqt2;
	}

	if (test == 4) {
		PVQuadTreeFlat<Vector1<entry> > *fqt1 = new PVQuadTreeFlat<Vector1<entry> >(0, MAX_VALUE, 0, MAX_VALUE, depth);
		std::cout << "sizeof(fqt1): " << sizeof(*fqt1) << std::endl;
		MEM_START(usage);
		BENCH_START(time);
		for(int i = 0; i < COUNT; ++i) {
			fqt1->insert(entries[i]);
		}
		BENCH_END(time, "PVQuadTreeFlat Vector1", COUNT, sizeof(entry), 1, 1);
		MEM_END(usage, "PVQuadTreeFlat Vector1");
		delete fqt1;
	}

	if (test == 5) {
		PVQuadTreeFlat<Vector2<entry> > *fqt2 = new PVQuadTreeFlat<Vector2<entry> >(0, MAX_VALUE, 0, MAX_VALUE, depth);
		std::cout << "sizeof(fqt2): " << sizeof(*fqt2) << std::endl;
		MEM_START(usage);
		BENCH_START(time);
		for(int i = 0; i < COUNT; ++i) {
			fqt2->insert(entries[i]);
		}
		BENCH_END(time, "PVQuadTreeFlat Vector2", COUNT, sizeof(entry), 1, 1);
		MEM_END(usage, "PVQuadTreeFlat Vector2");
		delete fqt2;
	}

#if 0
	if (tqt->compare(sqt)) {
		std::cout << "trees are equal" << std::endl;
	} else {
		std::cout << "trees differs" << std::endl;
	}
#endif

	return 0;
}
