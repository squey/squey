
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

void print_mem (const char *text, size_t s)
{
	double v = s / (1024. * 1024.);
	std::cout << text  << ": memory usage is: " << v << " Mib" << std::endl;
}

#define MAX_VALUE ((1<<22) - 1)

void usage()
{
	std::cout << "usage: test-quadtree entry-count" << std::endl;
	std::cout << std::endl;
}

#define TESTS_CHECK(vec, value) (std::find(vec.begin(), vec.end(), value) != vec.end())

// it's 8 because QuadTreeTmpl's size can not set
#define DEPTH 8

PVParallelView::PVHSVColor get_color(PVParallelView::PVHSVColor *colors)
{
	static int color_index = 0;
	PVParallelView::PVHSVColor c = colors[color_index];
	color_index = (color_index + 1) & 255;
	return c;
}

int main(int argc, char **argv)
{
	if (argc < 2) {
		usage();
		return 1;
	}

	int count = atoi(argv[1]);

	entry *entries = new entry [count];
	for(int i = 0; i < count; ++i) {
		entries[i].y1 = random() & MAX_VALUE;
		entries[i].y2 = random() & MAX_VALUE;
		entries[i].idx = i;
	}

	PVParallelView::PVHSVColor *colors = PVParallelView::PVHSVColor::init_colors(256);

	PVQuadTree<Vector1<entry>, entry> *sqt1 = new PVQuadTree<Vector1<entry>, entry>(0, MAX_VALUE, 0, MAX_VALUE, DEPTH);
	MEM_START(usage);
	BENCH_START(time);
	for(int i = 0; i < count; ++i) {
		sqt1->insert(entries[i]);
	}
	BENCH_END(time, "PVQuadTree", count, sizeof(entry), 1, 1);
	MEM_END(usage, "PVQuadTree");
	print_mem("PVQuadTree", sqt1->memory());

	std::vector<entry> res1;

	/* extraction of quadtree's data
	 */
	// res1.reserve(0);
	// BENCH_START(time_search);
	// sqt1->extract_first_y1y2(0, MAX_VALUE, 0, MAX_VALUE, res1);
	// BENCH_END(time_search, "PVQuadTree::serch", 1, 1, 1, 1);
	// std::cout << "search result size : " << res1.size() << std::endl;

	/* extraction of BCICode
	 */
	res1.clear();
	std::vector<PVParallelView::PVBCICode> codes;
	BENCH_START(time_extract);
	sqt1->extract_first_y1y2_bci(0, MAX_VALUE, 0, MAX_VALUE, codes);
	BENCH_END(time_extract, "BCICode Extraction", 1, 1, 1, 1);
	std::cout << "extraction result size : " << codes.size() << std::endl;

	if(sqt1) {
		delete sqt1;
	}

	return 0;
}
