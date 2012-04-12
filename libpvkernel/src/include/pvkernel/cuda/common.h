#ifndef PVCUDA_COMMON_H
#define PVCUDA_COMMON_H

#include <cuda.h>
#include <cuda_runtime.h>

#define picviz_verify_cuda(e) __picviz_verify_cuda(e, __FILE__, __LINE__)
#define __picviz_verify_cuda(e, F, L)\
		if ((e) != cudaSuccess) {\
			fprintf(stderr, "Cuda assert failed in %s:%d with %s.\n", F, L, cudaGetErrorString(cudaGetLastError()));\
			abort();\
		}
#define picviz_verify_cuda_kernel() __verify_cuda_kernel(__FILE__, __LINE__)
#define __verify_cuda_kernel(F, L) __picviz_verify_cuda(cudaGetLastError(), F, L)

namespace PVCuda {

void init_cuda();
int get_number_blocks();
size_t get_shared_mem_size();

}

#endif
