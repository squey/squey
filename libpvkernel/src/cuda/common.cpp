#include <pvkernel/cuda/common.h>
#include <stdio.h>

void PVCuda::init_cuda()
{
	cuInit(0);
	verify_cuda(cudaSetDevice(0)); // Tesla
	verify_cuda(cudaSetDeviceFlags(cudaDeviceMapHost));
}

int PVCuda::get_number_blocks()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	return prop.multiProcessorCount;
}

size_t PVCuda::get_shared_mem_size()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	return prop.sharedMemPerBlock;
}
