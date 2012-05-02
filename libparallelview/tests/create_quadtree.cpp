
#include <iostream>
#include <vector>
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>

#include <boost/random.hpp>
#include <boost/random/uniform_int_distribution.hpp>

#include <boost/math/distributions/normal.hpp>

#include <pvkernel/core/picviz_bench.h>

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

#define COUNT 100000000
//#define COUNT 10

#define MAX_VALUE ((1<<22) - 1)

int main(int argc, char **argv)
{
	if (argc != 2) {
		std::cout << "usage: test-quadtree tree-level" << std::endl;
		return 1;
	}

	int depth = atoi(argv[1]);
	long t1, t2;

	boost::mt19937 rnd(0);
	boost::random::uniform_int_distribution<unsigned> uni(0, UINT_MAX);

	PVQuadTree qt(0, MAX_VALUE, 0, MAX_VALUE, depth);

	entry *entries = new entry  [COUNT];
	for(int i = 0; i < COUNT; ++i) {
		entries[i].y1 = random() & MAX_VALUE;
		entries[i].y2 = random() & MAX_VALUE;
		entries[i].idx = i;
	}

	MEM_START(usage);
	BENCH_START(time);
	for(int i = 0; i < COUNT; ++i) {
		qt.insert(entries[i]);
	}
	BENCH_END(time, "time", COUNT, sizeof(entry), 1, 1);
	MEM_END(usage, "memory");

	// qt.dump();
	// qt.dump_stat();

	return 0;
}








