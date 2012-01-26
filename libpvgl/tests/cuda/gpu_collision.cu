#include <common/common.h>
#include <cuda/common.h>
#include <cuda/gpu_collision.h>

#include <cassert>
#include <cstdio>

#define NTHREADS_BLOCK 8
#define THREAD_CB_SIZE NB_INT_CB/NTHREADS_BLOCK
#define NB_INT_SHARED_BANK (48*1024/4)
#define verify_cuda(e) __verify_cuda(e, __FILE__, __LINE__)
#define __verify_cuda(e, F, L)\
		if ((e) != cudaSuccess) {\
			fprintf(stderr, "Cuda assert failed in %s:%d with %s.\n", F, L, cudaGetErrorString(cudaGetLastError()));\
			abort();\
		}
#define verify_cuda_kernel() __verify_cuda_kernel(__FILE__, __LINE__)

void __verify_cuda_kernel(const char* file, size_t line)
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "Kernel launch near %s:%d failed: %s.\n", file, line, cudaGetErrorString(err));
		abort();
	}
}

__global__ void cb_kernel_naive(Point* pts, size_t nlines, CollisionBuffer cb)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int size_grid = blockDim.x*gridDim.x;

	while (idx < nlines) {
		Point p = pts[idx];
		idx += size_grid;
		int idx_bit = p.y1*1024+p.y2;
		atomicOr(&cb[idx_bit>>5], ((int)1)<<(idx_bit&31));
	}
}

__global__ void cb_kernel(Point* pts, CollisionBuffer cb)
{
	// We need 3 shared memory banks of 48K
	__shared__ int shared_cb[NB_INT_SHARED_BANK];
	__shared__ int shared_cb2[NB_INT_SHARED_BANK];
	//__shared__ int shared_cb3[32*1024/4];

	//int* shared_cbs[] = {shared_cb, shared_cb2, shared_cb3};
	int* shared_cbs[] = {shared_cb, shared_cb2};
	int size_bank[] = {NB_INT_SHARED_BANK, NB_INT_SHARED_BANK, 32*1024/4};
	// First stage is to set shared_cb to 0
	// Each thread is responsible for setting NB_INT_CB/NTHREADS_BLOCK integers
	int idx_cb = threadIdx.x * THREAD_CB_SIZE;
	for (int i = idx_cb; i < idx_cb+THREAD_CB_SIZE; i++) {
		shared_cbs[i/(NB_INT_SHARED_BANK)][i] = 0;
	}
	__syncthreads();

	// Second stage is to compute the values of this block into shared_cb

	// Comptue the index of the point being process
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	// Get the point
	Point p = pts[idx];
	int idx_bit = p.y1*1024+p.y2;
	
	atomicOr(&shared_cbs[idx_bit/NB_INT_SHARED_BANK][idx_bit>>5], ((int)1)<<(idx_bit&31));

	__syncthreads();

	// Final stage is to merge shared_cb into the global cb
	for (int b = 0; b <= 2; b++) {
		int* bank = shared_cbs[b];
		int offset = b*NB_INT_SHARED_BANK;
		int sb = size_bank[b];
		for (int j = 0; j < sb; j++) {
			atomicOr(&cb[offset+j], bank[j]);
		}
	}
}

void gpu_c(Point* pts, size_t nlines, CollisionBuffer cb)
{
	assert(nlines % NTHREADS_BLOCK == 0);
	Point* device_pts;
	CollisionBuffer device_cb;

	verify_cuda(cudaMalloc(&device_pts, nlines*sizeof(Point)));
	verify_cuda(cudaMalloc(&device_cb, SIZE_CB));

	verify_cuda(cudaMemset(device_cb, 0x00, SIZE_CB));
	verify_cuda(cudaMemcpy(device_pts, pts, nlines*sizeof(Point), cudaMemcpyHostToDevice));

	cb_kernel_naive<<<get_number_blocks(),NTHREADS_BLOCK>>>(device_pts, nlines, device_cb);
	verify_cuda_kernel();

	verify_cuda(cudaMemcpy(cb, device_cb, SIZE_CB, cudaMemcpyDeviceToHost));

	verify_cuda(cudaFree(device_pts));
	verify_cuda(cudaFree(device_cb));
}
