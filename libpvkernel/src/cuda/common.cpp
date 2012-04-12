#include <pvkernel/cuda/common.h>
#include <stdio.h>

void PVCuda::init_cuda()
{
	cuInit(0);
	picviz_verify_cuda(cudaSetDevice(1)); // Tesla
	picviz_verify_cuda(cudaSetDeviceFlags(cudaDeviceMapHost));
}

int PVCuda::get_number_blocks()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 1);
	return prop.multiProcessorCount;
}

size_t PVCuda::get_shared_mem_size()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 1);
	return prop.sharedMemPerBlock;
}
