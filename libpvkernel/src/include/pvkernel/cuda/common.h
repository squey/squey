/**
 * \file common.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCUDA_COMMON_H
#define PVCUDA_COMMON_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <functional>

#define picviz_verify_cuda(e) __picviz_verify_cuda(e, __FILE__, __LINE__)
#define __picviz_verify_cuda(e, F, L)\
		if ((e) != cudaSuccess) {\
			fprintf(stderr, "Cuda assert failed in %s:%d with %s.\n", F, L, cudaGetErrorString(cudaGetLastError()));\
			abort();\
		}
#define picviz_verify_cuda_kernel() __verify_cuda_kernel(__FILE__, __LINE__)
#define __verify_cuda_kernel(F, L)\
	do {\
		int last_err = cudaGetLastError();\
		__picviz_verify_cuda(last_err, F, L);\
	} while(0);

namespace PVCuda {

void init_cuda();
void init_cuda_thread();
void init_gl_cuda();
int get_number_blocks();
size_t get_shared_mem_size();
size_t get_number_of_devices();
#ifndef __CUDACC__
void visit_usable_cuda_devices(std::function<void(int)> const& f);
#endif

}

#ifdef __CUDACC__
// nvcc does not support C++0x !
#define CUDA_CONSTEXPR const
#else
#define CUDA_CONSTEXPR constexpr
#endif

#endif
