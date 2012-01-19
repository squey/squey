#include "tbb/points_reduce.h"
#include "Point.h"

#include <iostream>
#include <cstdlib>

#include <tbb/tick_count.h>

void serial_c(Point* pts, size_t n, CollisionBuffer cb)
{
	for (size_t i = 0; i < n; i++) {
		int bit = (pts->y1)*1024 + (pts->y2);
		B_SET(cb[bit>>5], bit&31);
	}
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " size" << std::endl;
		return 1;
	}

	size_t nlines = atoll(argv[1]);
	Point* pts = allocate_buffer(nlines);

	fill_buffer(pts, nlines);

	CollisionBuffer cb_ref = allocate_CB();
	tbb::tick_count start = tbb::tick_count::now();
	serial_c(pts, nlines, cb_ref);
	tbb::tick_count end = tbb::tick_count::now();
	std::cout << "serial duration: " << (end-start).seconds() << std::endl;

	PointsReduce red(pts);
	start = tbb::tick_count::now();
	tbb::parallel_reduce(tbb::blocked_range<size_t>(0, nlines, 10000000), red, tbb::simple_partitioner());
	end = tbb::tick_count::now();

	std::cout << "parallel duration: " << (end-start).seconds() << std::endl;

	return 0;
}
