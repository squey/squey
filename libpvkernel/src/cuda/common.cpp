/**
 * \file common.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/cuda/common.h>
#include <stdio.h>

#define DEV_CUDA 0

void PVCuda::init_cuda()
{
	cuInit(0);
	init_cuda_thread();
}

void PVCuda::init_cuda_thread()
{
	picviz_verify_cuda(cudaSetDevice(DEV_CUDA)); // Tesla
	picviz_verify_cuda(cudaSetDeviceFlags(cudaDeviceMapHost));
}

size_t PVCuda::get_number_of_devices()
{
	int ret;
	picviz_verify_cuda(cudaGetDeviceCount(&ret));
	return ret;
}

void PVCuda::visit_usable_cuda_devices(std::function<void(int)> const& f)
{
	cudaDeviceProp prop;
	for (size_t i = 0; i < get_number_of_devices(); i++) {
		picviz_verify_cuda(cudaGetDeviceProperties(&prop, i));
		if (prop.major >= 2) {
			f(i);
		}
	}
}

void PVCuda::init_gl_cuda()
{
	cuInit(0);
	picviz_verify_cuda(cudaGLSetGLDevice(DEV_CUDA)); // Tesla
	picviz_verify_cuda(cudaSetDeviceFlags(cudaDeviceMapHost));
}

int PVCuda::get_number_blocks()
{
	int cur_device;
	picviz_verify_cuda(cudaGetDevice(&cur_device));
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, cur_device);
	return prop.multiProcessorCount;
}

size_t PVCuda::get_shared_mem_size()
{
	int cur_device;
	picviz_verify_cuda(cudaGetDevice(&cur_device));
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, cur_device);
	return prop.sharedMemPerBlock;
}
