#include <common/common.h>
#include <code_bz/types.h>
#include <code_bz/serial_bcodecb.h>
#include <code_bz/init.h>

#include <core-tbb/b_reduce.h>

#include <tbb/tick_count.h>

#include <iostream>
#include <cstdlib>

#include <string.h>

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " nb_codes" << std::endl;
		return 0;
	}

	std::cout << "BCodeCB size (bytes): " << NB_INT_BCODECB << std::endl;
	size_t ncodes = atoll(argv[1]);

	assert(sizeof(PVBCode) == sizeof(uint32_t));

	PVBCode* codes; // = (PVBCode*) malloc(ncodes*sizeof(PVBCode));
	posix_memalign((void**) &codes, 16, ncodes*sizeof(PVBCode));
	init_random_bcodes(codes, ncodes);

	BCodeCB serial_cb = allocate_BCodeCB();
	tbb::tick_count start = tbb::tick_count::now();
	serial_bcodecb(codes, ncodes, serial_cb);
	tbb::tick_count end = tbb::tick_count::now();

	std::cout << "Serial collision took " << (end-start).seconds() << "s." << std::endl;

	BCodeCB sse_cb = allocate_BCodeCB();
	start = tbb::tick_count::now();
	//sse_bcodecb(codes, ncodes, sse_cb);
	end = tbb::tick_count::now();

	std::cout << "SSE collision took " << (end-start).seconds() << "s." << std::endl;
	std::cout << "memcmp serial vs. sse: " << (memcmp(sse_cb, serial_cb, SIZE_BCODECB) == 0) << std::endl;

	BCodeReduce tbb_red(codes);
	start = tbb::tick_count::now();
	tbb::parallel_reduce(tbb::blocked_range<size_t>(0, ncodes, 10000000), tbb_red, tbb::simple_partitioner());
	end = tbb::tick_count::now();

	std::cout << "TBB collision took " << (end-start).seconds() << "s." << std::endl;

	std::cout << "memcmp serial vs. tbb: " << (memcmp(tbb_red.cb(), serial_cb, SIZE_BCODECB) == 0) << std::endl;

	return 0;
}
