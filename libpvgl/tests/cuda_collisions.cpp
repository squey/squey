#include <common/common.h>
#include <common/bench.h>
#include <code_bz/serial_bcodecb.h>
#include <cuda/common.h>
#include <cuda/gpu_bccb.h>
#include <code_bz/init.h>

#include <iostream>
#include <algorithm>

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

	size_t ncodes = atoll(argv[1]);
	PVBCode* codes;
	posix_memalign((void**) &codes, 16, sizeof(PVBCode)*ncodes);
	init_random_bcodes(codes, ncodes);

	//std::sort((uint32_t*) codes, (uint32_t*)codes+ncodes);

	BCodeCB cb_ref = allocate_BCodeCB();
	BENCH_START(serial);
	serial_bcodecb(codes, ncodes, cb_ref);
	BENCH_END(serial, "serial collision", ncodes, sizeof(PVBCode), 1, SIZE_BCODECB);

	BCodeCB cb_gpu = allocate_BCodeCB();
	tbb::tick_count start = tbb::tick_count::now();
	gpu_bccb_2dim(codes, ncodes, cb_gpu);
	tbb::tick_count end = tbb::tick_count::now();
	std::cout << "GPU duration: " << (end-start).seconds() << " s" << std::endl;

	// Combining two
	std::cout << "memcmp serial vs. parallel: " << (memcmp(cb_ref, cb_gpu, SIZE_BCODECB) == 0) << std::endl;

	write(4, cb_ref, SIZE_BCODECB);
	write(5, cb_gpu, SIZE_BCODECB);

	free(cb_ref);
	free(cb_gpu);

	return 0;
}
