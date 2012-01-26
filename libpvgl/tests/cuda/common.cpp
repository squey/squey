#include <cuda/common.h>

void init_cuda()
{
	cudaSetDevice(0); // Tesla
}

int get_number_blocks()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	return prop.multiProcessorCount * 2;
}
