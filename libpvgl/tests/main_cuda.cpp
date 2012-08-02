/**
 * \file main_cuda.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <core-tbb/points_reduce.h>
#include <common/common.h>
#include <common/serial_cb.h>
#include <cuda/common.h>
#include <cuda/gpu_collision.h>

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
	init_cuda();

	size_t nlines = atoll(argv[1]);
	Point* pts = allocate_buffer(nlines);

	fill_buffer(pts, nlines);

	CollisionBuffer cb_ref = allocate_CB();
	tbb::tick_count start = tbb::tick_count::now();
	//serial_cb(pts, nlines, cb_ref);
	tbb::tick_count end = tbb::tick_count::now();
	std::cout << "serial duration: " << (end-start).seconds() << std::endl;

	CollisionBuffer cb_gpu = allocate_CB();
	start = tbb::tick_count::now();
	gpu_c(pts, nlines, cb_gpu);
	end = tbb::tick_count::now();
	std::cout << "GPU duration: " << (end-start).seconds() << std::endl;

	std::cout << "memcmp serial vs. parallel: " << (memcmp(cb_ref, cb_gpu, SIZE_CB) == 0) << std::endl;

	write(4, cb_ref, SIZE_CB);
	write(5, cb_gpu, SIZE_CB);

	free_CB(cb_ref);
	free_CB(cb_gpu);

	return 0;
}
