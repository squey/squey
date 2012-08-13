/**
 * \file common.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <cuda/common.h>
#include <stdio.h>

void init_cuda()
{
	cuInit(0);
	verify_cuda(cudaSetDevice(0)); // Tesla
	verify_cuda(cudaSetDeviceFlags(cudaDeviceMapHost));
}

int get_number_blocks()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	return prop.multiProcessorCount;
}

size_t get_shared_mem_size()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	return prop.sharedMemPerBlock;
}
