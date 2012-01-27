#include <core-tbb/points_reduce.h>
#include <common/common.h>
#include <common/serial_cb.h>

#include <iostream>
#include <cstdlib>
#include <string.h>

#include <tbb/tick_count.h>

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
	serial_cb(pts, nlines, cb_ref);
	tbb::tick_count end = tbb::tick_count::now();
	std::cout << "serial duration: " << (end-start).seconds() << std::endl;

	PointsReduce red(pts);
	start = tbb::tick_count::now();
	tbb::parallel_reduce(tbb::blocked_range<size_t>(0, nlines, 100000), red, tbb::simple_partitioner());
	end = tbb::tick_count::now();

	std::cout << "parallel duration: " << (end-start).seconds() << std::endl;

	std::cout << "memcmp serial vs. parallel: " << (memcmp(cb_ref, red.cb(), SIZE_CB) == 0) << std::endl;

	return 0;
}
