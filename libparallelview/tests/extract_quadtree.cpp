
#include <iostream>
#include <vector>
#include <algorithm>

#include <stdint.h>
#include <limits.h>
#include <stdlib.h>

#include <boost/random.hpp>
#include <boost/random/uniform_int_distribution.hpp>

#include <boost/math/distributions/normal.hpp>

#include <pvkernel/core/picviz_bench.h>

#include <pvparallelview/PVHSVColor.h>
#include <pvparallelview/PVBCICode.h>

// gros hack pour que les quadtree connaissent la structure entry

#pragma pack(push)
#pragma pack(4)

struct entry {
	uint32_t y1, y2;
	uint32_t idx;

	bool operator==(const entry &e)
	{
		return ((y1 == e.y1) || (y2 == e.y2) || (idx == e.idx));
	}
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

#define MAX_VALUE ((1<<22) - 1)


void print_mem (const char *text, size_t s)
{
	double v = s / (1024. * 1024.);
	std::cout << text  << ": memory usage is: " << v << " Mib" << std::endl;
}

void usage()
{
	std::cout << "usage: test-quadtree entry-count what" << std::endl;
	std::cout << std::endl;
	std::cout << "what can be:" << std::endl;
	std::cout << "  0: PVQuadTree::extract_first_y1y2 with full area" << std::endl;
	std::cout << "  1: PVQuadTree::extract_first_y1y2_bci with full area" << std::endl;
	std::cout << "  2: PVQuadTree::extract_subtree_y1 with full area" << std::endl;
	std::cout << "  3: PVQuadTree::extract_subtree_y1 with half area" << std::endl;
	std::cout << "  4: PVQuadTree::extract_subtree_y1y2 with quarter area" << std::endl;
	std::cout << "  5: PVQuadTree::extract_subtree_y1y2 with quarter area for each quarter" << std::endl;
	std::cout << std::endl;
}

// it's 8 because QuadTreeTmpl's size can not set
#define DEPTH 8

int main(int argc, char **argv)
{
	if (argc != 3) {
		usage();
		return 1;
	}

	int count = atoi(argv[1]);
	int what = atoi(argv[2]);

	if(what > 5) {
		usage();
		return 2;
	}

	entry *entries = new entry [count];
	for(int i = 0; i < count; ++i) {
		entries[i].y1 = random() & MAX_VALUE;
		entries[i].y2 = random() & MAX_VALUE;
		entries[i].idx = i;
	}

	PVQuadTree<Vector1<entry>, entry> *sqt1 = new PVQuadTree<Vector1<entry>, entry>(0, MAX_VALUE, 0, MAX_VALUE, DEPTH);
	std::cout << "Filling quadtree, it can take a while..." << std::endl;
	for(int i = 0; i < count; ++i) {
		sqt1->insert(entries[i]);
	}

	std::vector<entry> res1;

	/* extraction of quadtree's raw data
	 */
	if(what == 0) {
		res1.reserve(0);
		BENCH_START(time_search);
		sqt1->extract_first_y1y2(0, MAX_VALUE, 0, MAX_VALUE, res1);
		BENCH_END(time_search, "PVQuadTree::serch", 1, 1, 1, 1);
		std::cout << "search result size : " << res1.size() << std::endl;
	}

	/* extraction of BCICode
	 */
	if(what == 1) {
		std::vector<PVParallelView::PVBCICode> codes;
		BENCH_START(time_extract);
		sqt1->extract_first_y1y2_bci(0, MAX_VALUE, 0, MAX_VALUE, codes);
		BENCH_END(time_extract, "BCICode Extraction", 1, 1, 1, 1);
		std::cout << "extraction result size : " << codes.size() << std::endl;
	}

	/* comparison with a full extraction
	 */
	if(what == 2) {
		PVQuadTree<Vector1<entry>, entry> *subtree;
		BENCH_START(time_extract_subtree);
		subtree = sqt1->extract_subtree_y1(0, MAX_VALUE);
		BENCH_END(time_extract_subtree, "Subtree Extraction", 1, 1, 1, 1);
		print_mem("QuadTree", sqt1->memory());
		print_mem("SubQuadTree", subtree->memory());
		if(*sqt1 == *subtree) {
			std::cout << "subtree is equal" << std::endl;
		} else {
			std::cout << "subtree is different" << std::endl;
		}
		delete subtree;
	}

	if(what == 3) {
		PVQuadTree<Vector1<entry>, entry> *subtree;
		BENCH_START(time_extract_subtree);
		subtree = sqt1->extract_subtree_y1(0, MAX_VALUE >> 1);
		BENCH_END(time_extract_subtree, "Subtree Extraction", 1, 1, 1, 1);
		print_mem("QuadTree", sqt1->memory());
		print_mem("SubQuadTree", subtree->memory());
		std::cout << "QuadTree's elements count: " << sqt1->elements() << std::endl;
		std::cout << "SubQuadTree's elements count: " << subtree->elements() << std::endl;
		delete subtree;
	}


	if(what == 4) {
		PVQuadTree<Vector1<entry>, entry> *subtree;
		BENCH_START(time_extract_subtree);
		subtree = sqt1->extract_subtree_y1y2(0, (MAX_VALUE >> 1),
		                                     0, (MAX_VALUE >> 1));
		BENCH_END(time_extract_subtree, "Subtree Extraction", 1, 1, 1, 1);
		print_mem("QuadTree", sqt1->memory());
		print_mem("SubQuadTree", subtree->memory());
		std::cout << "QuadTree's elements count: " << sqt1->elements() << std::endl;
		std::cout << "SubQuadTree's elements count: " << subtree->elements() << std::endl;
		// sqt1->dump(std::cout);
		// subtree->dump(std::cerr);
		delete subtree;
	}

	if(what == 5) {
		{
			// SW quarter
			PVQuadTree<Vector1<entry>, entry> *subtree;
			BENCH_START(time_extract_subtree);
			subtree = sqt1->extract_subtree_y1y2(0, MAX_VALUE >> 1,
			                                     0, MAX_VALUE >> 1);
			BENCH_END(time_extract_subtree, "Subtree Extraction", 1, 1, 1, 1);
			print_mem("QuadTree", sqt1->memory());
			print_mem("SubQuadTree", subtree->memory());
			std::cout << "QuadTree's elements count: " << sqt1->elements() << std::endl;
			std::cout << "SubQuadTree's elements count: " << subtree->elements() << std::endl;
			delete subtree;
		}

		{
			// SE quarter
			PVQuadTree<Vector1<entry>, entry> *subtree;
			BENCH_START(time_extract_subtree);
			subtree = sqt1->extract_subtree_y1y2(             0, MAX_VALUE >> 1,
			                                     MAX_VALUE >> 1, MAX_VALUE);
			BENCH_END(time_extract_subtree, "Subtree Extraction", 1, 1, 1, 1);
			print_mem("QuadTree", sqt1->memory());
			print_mem("SubQuadTree", subtree->memory());
			std::cout << "QuadTree's elements count: " << sqt1->elements() << std::endl;
			std::cout << "SubQuadTree's elements count: " << subtree->elements() << std::endl;
			delete subtree;
		}

		{
			// NE quarter
			PVQuadTree<Vector1<entry>, entry> *subtree;
			BENCH_START(time_extract_subtree);
			subtree = sqt1->extract_subtree_y1y2(MAX_VALUE >> 1, MAX_VALUE,
			                                     MAX_VALUE >> 1, MAX_VALUE);
			BENCH_END(time_extract_subtree, "Subtree Extraction", 1, 1, 1, 1);
			print_mem("QuadTree", sqt1->memory());
			print_mem("SubQuadTree", subtree->memory());
			std::cout << "QuadTree's elements count: " << sqt1->elements() << std::endl;
			std::cout << "SubQuadTree's elements count: " << subtree->elements() << std::endl;
			delete subtree;
		}

		{
			// NW quarter
			PVQuadTree<Vector1<entry>, entry> *subtree;
			BENCH_START(time_extract_subtree);
			subtree = sqt1->extract_subtree_y1y2(MAX_VALUE >> 1, MAX_VALUE,
			                                                  0, MAX_VALUE >> 1);
			BENCH_END(time_extract_subtree, "Subtree Extraction", 1, 1, 1, 1);
			print_mem("QuadTree", sqt1->memory());
			print_mem("SubQuadTree", subtree->memory());
			std::cout << "QuadTree's elements count: " << sqt1->elements() << std::endl;
			std::cout << "SubQuadTree's elements count: " << subtree->elements() << std::endl;
			delete subtree;
		}

	}

	if(sqt1) {
		delete sqt1;
	}

	return 0;
}
