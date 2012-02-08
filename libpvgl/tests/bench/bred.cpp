#include <common/common.h>
#include <common/bench.h>

#include <code_bz/types.h>
#include <code_bz/serial_bcodecb.h>
#include <code_bz/init.h>

#include <core-tbb/b_reduce.h>

#include <iostream>
#include <algorithm>
#include <set>
#include <cstdlib>

#include <QSet>

#include <omp.h>

#include <string.h>

#include <pvkernel/core/PVAllocators.h>

#define COLLISION_BENCH_END(name, desc, cb)\
	BENCH_END(name, desc, ncodes, sizeof(PVBCode), NB_INT_BCODECB, sizeof(int));\
	CHECK(memcmp(cb, cb_ref, SIZE_BCODECB) == 0);

#define LAUNCH_BENCH(name, desc, f)\
	BENCH_START(name);\
	f(codes, ncodes, cb);\
	COLLISION_BENCH_END(name, desc, cb);

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " nb_codes" << std::endl;
		return 0;
	}

	// Bootstrap omp
	int a = 0;
#pragma omp parallel for
	for (int i = 0; i < 1000000; i++) {
		a += i;
	}
	std::cout << "BCodeCB size (bytes): " << NB_INT_BCODECB << std::endl;
	size_t ncodes = atoll(argv[1]);

	assert(sizeof(PVBCode) == sizeof(uint32_t));

	PVBCode_ap codes;
	posix_memalign((void**) &codes, 16, ncodes*sizeof(PVBCode));
	//init_constant_bcodes(codes, ncodes);
	init_random_bcodes(codes, ncodes);

	BCodeCB cb_ref = allocate_BCodeCB();
	BCodeCB cb = allocate_BCodeCB();
	
	uint32_t* tiles_cb[NTILE_CB];
	for (unsigned int i = 0; i < NTILE_CB; i++) {
		posix_memalign((void**) &tiles_cb[i], 16, TILE_SIZE_INT*sizeof(uint32_t));
		memset(tiles_cb[i], 0, TILE_SIZE_INT*sizeof(uint32_t));
	}
	BENCH_START(serial);
	serial_bcodecb(codes, ncodes, cb_ref);
	BENCH_END(serial, "serial collision", ncodes, sizeof(PVBCode), NB_INT_BCODECB, sizeof(int));

	QSet<uint32_t> set_reds;
	set_reds.reserve(ncodes);
	BENCH_START(set);
	for (size_t i = 0; i < ncodes; i++) {
		set_reds.insert(codes[i].int_v);
	}
	COLLISION_BENCH_END(set, "set", cb_ref);

	BENCH_START(tile);
	//bcodecb_tile(codes, ncodes, cb, tiles_cb);
	BENCH_END(tile, "tile collision", ncodes, sizeof(PVBCode), NB_INT_BCODECB, sizeof(int));

	LAUNCH_BENCH(branch, "branch serial collision", bcodecb_branch);
	LAUNCH_BENCH(sse, "sse serial collision", bcodecb_sse);
	LAUNCH_BENCH(sse_branch, "sse-branch serial collision", bcodecb_sse_branch);
	LAUNCH_BENCH(sse_branch2, "sse-branch2 serial collision", bcodecb_sse_branch2);

	LAUNCH_BENCH(sse_branch_omp, "sse-branch omp collision", omp_bcodecb_sse_branch);

	// TODO: bootstrap TBB ?
	// TBB
	{
		BCodeReduce tbb_red(codes);
		BENCH_START(btbb);
		tbb::parallel_reduce(tbb::blocked_range<size_t>(0, ncodes, 10000000), tbb_red, tbb::simple_partitioner());
		COLLISION_BENCH_END(btbb, "tbb collision", tbb_red.cb());
	}
	{
		BCodeReduce tbb_red(codes);
		BENCH_START(btbb);
		tbb::parallel_reduce(tbb::blocked_range<size_t>(0, ncodes, 10000000), tbb_red, tbb::simple_partitioner());
		COLLISION_BENCH_END(btbb, "tbb collision", tbb_red.cb());
	}

	std::cout << "Same w/ the codes sorted..." << std::endl;
	std::sort((uint32_t*) codes, ((uint32_t*) codes)+ncodes);
	std::cout << "Codes sorted." << std::endl;
	{
		BENCH_START(serial);
		serial_bcodecb(codes, ncodes, cb_ref);
		BENCH_END(serial, "serial collision", ncodes, sizeof(PVBCode), NB_INT_BCODECB, sizeof(int));
		LAUNCH_BENCH(branch, "branch serial collision", bcodecb_branch);
		LAUNCH_BENCH(sse, "sse serial collision", bcodecb_sse);
		LAUNCH_BENCH(sse_branch, "sse-branch serial collision", bcodecb_sse_branch);
		LAUNCH_BENCH(sse_branch2, "sse-branch2 serial collision", bcodecb_sse_branch2);
	}
	
	std::cout << "Same w/ always the same code..." << std::endl;
	{
		init_constant_bcodes(codes, ncodes);
		BENCH_START(serial);
		serial_bcodecb(codes, ncodes, cb_ref);
		BENCH_END(serial, "serial collision", ncodes, sizeof(PVBCode), NB_INT_BCODECB, sizeof(int));
		LAUNCH_BENCH(branch, "branch serial collision", bcodecb_branch);
		LAUNCH_BENCH(sse, "sse serial collision", bcodecb_sse);
		LAUNCH_BENCH(sse_branch, "sse-branch serial collision", bcodecb_sse_branch);
		LAUNCH_BENCH(sse_branch2, "sse-branch2 serial collision", bcodecb_sse_branch2);
	}

	return 0;
}
