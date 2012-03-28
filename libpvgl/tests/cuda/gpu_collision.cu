#include <common/common.h>
#include <cuda/common.h>
#include <cuda/gpu_collision.h>
#include <tbb/tick_count.h>
#include <iostream>

#include <cassert>
#include <cstdio>

#define NTHREADS_BLOCK 1024
#define NBANKS 32
#define NB_INT_SHARED_CB_THREAD NB_INT_CB/NTHREADS_BLOCK

#define verify(e) __verify(e, __FILE__, __LINE__)
#define __verify(e, F, L)\
	if (!(e)) {\
		fprintf(stderr, "valid assertion failed at %s:%d: %s.\n", F, L, #e);\
		abort();\
	}
		
void __verify_cuda_kernel(const char* file, size_t line)
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "Kernel launch near %s:%d failed: %s.\n", file, line, cudaGetErrorString(err));
		abort();
	}
}

__global__ void cb_kernel_naive(int2* pts, size_t nlines, CollisionBuffer cb)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int size_grid = blockDim.x*gridDim.x;

	while (idx < nlines) {
		int2 p = pts[idx];
		idx += size_grid;
		int idx_bit = p.x*1024+p.y;
		atomicOr(&cb[idx_bit>>5], ((int)1)<<(idx_bit&31));
	}
}

__global__ void cb_kernel(int2* pts, size_t nlines, CollisionBuffer cb)
{
	__shared__ int shared_cb[NB_INT_CB];

	// First stage is to set shared_cb to 0
	for (int i = threadIdx.x; i < NB_INT_CB; i += blockDim.x) {
		shared_cb[i] = 0;
	}

	__syncthreads();

	// Second stage is to compute the values of this block into shared_cb

	// Comptue the index of the point being process
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int size_grid = blockDim.x*gridDim.x;

	while (idx < nlines) {
		// Get the point
		int2 p = pts[idx];
		idx += size_grid;
		int idx_bit = (p.x)*PIXELS_CB+(p.y);
		// This atomic operation is the cause of branch divergances !
		atomicOr(&shared_cb[idx_bit>>5], ((int)1)<<(idx_bit&31));
	}

	__syncthreads();

	// Final stage is to merge shared_cb into the global cb
	for (int i = threadIdx.x; i < NB_INT_CB; i += blockDim.x) {
		atomicOr(&cb[i], shared_cb[i]);
	}
}

__global__ void cb_kernel_unrolled(int2* pts, size_t nlines, CollisionBuffer cb)
{
	__shared__ int shared_cb[NB_INT_CB];

	// First stage is to set shared_cb to 0
	for (int i = threadIdx.x; i < NB_INT_CB; i += blockDim.x) {
		shared_cb[i] = 0;
	}

	__syncthreads();

	// Second stage is to compute the values of this block into shared_cb

	// Comptue the index of the point being process
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int size_grid = blockDim.x*gridDim.x;

	// Unroll this loop by 2*4x
	while (idx < nlines) {
		// Get the point
		int2 p = pts[idx];
		int2 p1 = pts[idx + size_grid];
		int2 p2 = pts[idx + 2*size_grid];
		int2 p3 = pts[idx + 3*size_grid];
		idx += 4*size_grid;
		
		int idx_bit = (p.x)*PIXELS_CB+(p.y);
		int idx_bit1 = (p1.x)*PIXELS_CB+(p1.y);
		int idx_bit2 = (p2.x)*PIXELS_CB+(p2.y);
		int idx_bit3 = (p3.x)*PIXELS_CB+(p3.y);

		// These atomics operations are the cause of branch divergances !
		atomicOr(&shared_cb[idx_bit>>5], ((int)1)<<(idx_bit&31));
		atomicOr(&shared_cb[idx_bit1>>5], ((int)1)<<(idx_bit1&31));
		atomicOr(&shared_cb[idx_bit2>>5], ((int)1)<<(idx_bit2&31));
		atomicOr(&shared_cb[idx_bit3>>5], ((int)1)<<(idx_bit3&31));
		
		// Get the point
		p = pts[idx];
		p1 = pts[idx + size_grid];
		p2 = pts[idx + 2*size_grid];
		p3 = pts[idx + 3*size_grid];
		idx += 4*size_grid;
		
		idx_bit = (p.x)*PIXELS_CB+(p.y);
		idx_bit1 = (p1.x)*PIXELS_CB+(p1.y);
		idx_bit2 = (p2.x)*PIXELS_CB+(p2.y);
		idx_bit3 = (p3.x)*PIXELS_CB+(p3.y);

		// These atomics operations are the cause of branch divergances !
		atomicOr(&shared_cb[idx_bit>>5], ((int)1)<<(idx_bit&31));
		atomicOr(&shared_cb[idx_bit1>>5], ((int)1)<<(idx_bit1&31));
		atomicOr(&shared_cb[idx_bit2>>5], ((int)1)<<(idx_bit2&31));
		atomicOr(&shared_cb[idx_bit3>>5], ((int)1)<<(idx_bit3&31));
	}

	__syncthreads();

	// Final stage is to merge shared_cb into the global cb
	for (int i = threadIdx.x; i < NB_INT_CB; i += blockDim.x) {
		atomicOr(&cb[i], shared_cb[i]);
	}
}

void gpu_c(Point* pts, size_t nlines, CollisionBuffer cb)
{
	Point* device_pts;
	CollisionBuffer device_cb;

	verify_cuda(cudaMalloc(&device_pts, nlines*sizeof(Point)));
	verify_cuda(cudaMalloc(&device_cb, SIZE_CB));
	verify((uintptr_t)device_pts % sizeof(int2) == 0);

	verify_cuda(cudaMemset(device_cb, 0x00, SIZE_CB));
	verify_cuda(cudaMemcpy(device_pts, pts, nlines*sizeof(Point), cudaMemcpyHostToDevice));

	cudaEvent_t start,end;
	verify_cuda(cudaEventCreate(&start));
	verify_cuda(cudaEventCreate(&end));

	int nblocks = get_number_blocks()*2;
	int size_grid = nblocks*NTHREADS_BLOCK;
	verify(nlines % 8*size_grid == 0);
	verify(nlines > 8*size_grid);
	fprintf(stderr, "Grid size is %d.\n", size_grid);

	verify_cuda(cudaEventRecord(start, 0));
	cb_kernel_unrolled<<<nblocks,NTHREADS_BLOCK>>>((int2*) device_pts, nlines, device_cb);
	verify_cuda_kernel();
	verify_cuda(cudaEventRecord(end, 0));
	verify_cuda(cudaEventSynchronize(end));

	verify_cuda(cudaMemcpy(cb, device_cb, SIZE_CB, cudaMemcpyDeviceToHost));

	verify_cuda(cudaFree(device_pts));
	verify_cuda(cudaFree(device_cb));

	float time = 0;
	verify_cuda(cudaEventElapsedTime(&time, start, end));

	fprintf(stderr, "CUDA kernel time (%d blocks): %0.4f ms, BW: %0.4f MB/s\n", nblocks, time, ((float)(nlines*sizeof(Point)+SIZE_CB*nblocks))/(1024.0f*1024.0f*(time/1000.0f)));
}
