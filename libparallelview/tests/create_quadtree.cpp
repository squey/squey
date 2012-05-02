
#include <iostream>
#include <vector>
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>

#include <sys/time.h>

#include <boost/random.hpp>
#include <boost/random/uniform_int_distribution.hpp>

#include <boost/math/distributions/normal.hpp>

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

long tick()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec * 1000) + (tv.tv_usec / 1000);
}

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

	t1 = tick();
	for(int i = 0; i < COUNT; ++i) {
		qt.insert(entries[i]);
	}
	t2 = tick();

	// std::cout << depth << " " << t2 - t1 << std::endl;
	 std::cout << "insert " << COUNT << " element(s) of size " << sizeof(entry) << " in " << t2 - t1 << " ms" << std::endl;
	// qt.dump();
	// qt.dump_stat();

	return 0;
}








