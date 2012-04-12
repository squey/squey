#ifndef PVCUDA_COMMON_H
#define PVCUDA_COMMON_H

#include <cuda.h>
#include <cuda_runtime.h>

#define verify_cuda(e) __verify_cuda(e, __FILE__, __LINE__)
#define __verify_cuda(e, F, L)\
		if ((e) != cudaSuccess) {\
			fprintf(stderr, "Cuda assert failed in %s:%d with %s.\n", F, L, cudaGetErrorString(cudaGetLastError()));\
			abort();\
		}
#define verify_cuda_kernel() __verify_cuda_kernel(__FILE__, __LINE__)

namespace PVCuda {

void init_cuda();
int get_number_blocks();
size_t get_shared_mem_size();

}

#endif
