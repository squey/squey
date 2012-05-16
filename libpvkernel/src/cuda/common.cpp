#include <pvkernel/cuda/common.h>
#include <stdio.h>

#define DEV_CUDA 0

void PVCuda::init_cuda()
{
	cuInit(0);
	picviz_verify_cuda(cudaSetDevice(DEV_CUDA)); // Tesla
	picviz_verify_cuda(cudaSetDeviceFlags(cudaDeviceMapHost));
}

void PVCuda::init_gl_cuda()
{
	cuInit(0);
	picviz_verify_cuda(cudaGLSetGLDevice(DEV_CUDA)); // Tesla
	picviz_verify_cuda(cudaSetDeviceFlags(cudaDeviceMapHost));
}

int PVCuda::get_number_blocks()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, DEV_CUDA);
	return prop.multiProcessorCount;
}

size_t PVCuda::get_shared_mem_size()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, DEV_CUDA);
	return prop.sharedMemPerBlock;
}
