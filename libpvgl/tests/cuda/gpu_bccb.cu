/**
 * \file gpu_bccb.cu
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <common/common.h>
#include <common/bench.h>

#include <cuda/common.h>
#include <cuda/gpu_bccb.h>

#define NTHREADS_BLOCK 1024
// 48K of shared memory
#define NB_INT_SHARED ((48*1024)/sizeof(unsigned int))
// Take 'n' codes into shared memory, and the rest for the temporary shared cb
#define NB_CODES_SHARED 4
#define NB_INT_CB_SHARED (NB_INT_SHARED-NB_CODES_SHARED)
//#define NB_INT_SHARED ((NB_INT_SHARED_MEM*NB_INT_BCODECB)/NB_INT_BCODECB)
#define NB_PASSES ((NB_INT_BCODECB+NB_INT_SHARED-1)/NB_INT_SHARED)
//#define NB_PASSES ((NB_INT_BCODECB)/NB_INT_SHARED)
#define NB_INT_SHARED_LAST_PASS ((NB_INT_BCODECB)%(NB_INT_SHARED))
#define L2_CACHE (768*1024)
#define L2_CACHE_INT (L2_CACHE/sizeof(int))

#if (NB_BCODE/(8*4)) % ((48*1024)/(4)) != 0
#define LAST_PASS_DIFFERENT
#else
#undef NB_INT_SHARED_LAST_PASS
#define NB_INT_SHARED_LAST_PASS NB_INT_SHARED
#endif

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

__global__ void bccb_kernel_naive(unsigned int* codes, size_t n, unsigned int* cb)
{
	//int idx = threadIdx.x + blockIdx.x * blockDim.x;
	//int size_grid = blockDim.x*gridDim.x;
	int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;
	int size_grid = blockDim.x*gridDim.x*gridDim.y;

	while (idx < n) {
		unsigned int p = codes[idx];
		idx += size_grid;
		atomicOr(&cb[p>>5], ((unsigned int)1)<<(p&31));
	}
}

__global__ void bccb_kernel_hybrid(unsigned int* codes, size_t n, unsigned int* cb)
{
	__shared__ int shared_cb[NB_INT_SHARED];

	// First stage is to set shared_cb to 0
	for (int i = threadIdx.x; i < NB_INT_SHARED; i += blockDim.x) {
		shared_cb[i] = 0;
	}

	__syncthreads();

	int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;
	int size_grid = blockDim.x*gridDim.x*gridDim.y;

	// blockIdx.y gives the pass that we must care about
	int idx_start_scb = blockIdx.y*NB_INT_SHARED;
	int idx_end_scb = idx_start_scb + NB_INT_SHARED;

	for (; idx < n; idx += size_grid) {
		int p1 = codes[idx];
		const int idx_scb1 = p1>>5;
		const unsigned int rem1 = ((unsigned int)1)<<(p1&31);

		if ((idx_scb1 >= idx_start_scb) && (idx_scb1 < idx_end_scb)) {
			atomicOr(&shared_cb[idx_scb1 - idx_start_scb], rem1);
		}
		else {
			atomicOr(&cb[idx_scb1], rem1);
		}
	}

	int nb_int_shared;
	if (blockIdx.y == gridDim.y-1) {
		// Last pass has less integer into the global buffer
		nb_int_shared = NB_INT_SHARED_LAST_PASS;
	}
	else {
		nb_int_shared = NB_INT_SHARED;
	}
	__syncthreads();

	// Merge shared_cb into the global cb
	for (int j = threadIdx.x; j < nb_int_shared; j += blockDim.x) {
		atomicOr(&cb[idx_start_scb+j], shared_cb[j]);
	}
}

__global__ void bccb_kernel_hybrid_unrolled(unsigned int* codes, size_t n, unsigned int* cb)
{
	__shared__ int shared_cb[NB_INT_SHARED];

	// First stage is to set shared_cb to 0
	for (int i = threadIdx.x; i < NB_INT_SHARED; i += blockDim.x) {
		shared_cb[i] = 0;
	}

	__syncthreads();

	int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;
	int size_grid = blockDim.x*gridDim.x*gridDim.y;

	// blockIdx.y gives the pass that we must care about
	int idx_start_scb = blockIdx.y*NB_INT_SHARED;
	int idx_end_scb = idx_start_scb + NB_INT_SHARED;

#pragma unroll 4
	for (; idx < n; idx += 4*size_grid) {
		int p1 = codes[idx];
		int p2 = codes[idx+size_grid];
		int p3 = codes[idx+2*size_grid];
		int p4 = codes[idx+3*size_grid];

		const int idx_scb1 = p1>>5;
		const int idx_scb2 = p2>>5;
		const int idx_scb3 = p3>>5;
		const int idx_scb4 = p4>>5;

		const unsigned int rem1 = ((unsigned int)1)<<(p1&31);
		const unsigned int rem2 = ((unsigned int)1)<<(p2&31);
		const unsigned int rem3 = ((unsigned int)1)<<(p3&31);
		const unsigned int rem4 = ((unsigned int)1)<<(p4&31);

		if ((idx_scb1 >= idx_start_scb) && (idx_scb1 < idx_end_scb)) {
			atomicOr(&shared_cb[idx_scb1 - idx_start_scb], rem1);
		}
		else {
			atomicOr(&cb[idx_scb1], rem1);
		}

		if ((idx_scb2 >= idx_start_scb) && (idx_scb2 < idx_end_scb)) {
			atomicOr(&shared_cb[idx_scb2 - idx_start_scb], rem2);
		}
		else {
			atomicOr(&cb[idx_scb2], rem2);
		}

		if ((idx_scb3 >= idx_start_scb) && (idx_scb3 < idx_end_scb)) {
			atomicOr(&shared_cb[idx_scb3 - idx_start_scb], rem3);
		}
		else {
			atomicOr(&cb[idx_scb3], rem3);
		}

		if ((idx_scb4 >= idx_start_scb) && (idx_scb4 < idx_end_scb)) {
			atomicOr(&shared_cb[idx_scb4 - idx_start_scb], rem4);
		}
		else {
			atomicOr(&cb[idx_scb4], rem4);
		}
	}

	
#ifdef LAST_PASS_DIFFERENT
	int nb_int_shared;
	if (blockIdx.y == gridDim.y-1) {
		// Last pass has less integer into the global buffer
		nb_int_shared = NB_INT_SHARED_LAST_PASS;
	}
	else {
		nb_int_shared = NB_INT_SHARED;
	}
	__syncthreads();

	// Merge shared_cb into the global cb
	for (int j = threadIdx.x; j < nb_int_shared; j += blockDim.x) {
		atomicOr(&cb[idx_start_scb+j], shared_cb[j]);
	}
#else
	__syncthreads();

	// Merge shared_cb into the global cb
	for (int j = threadIdx.x; j < NB_INT_SHARED; j += blockDim.x) {
		atomicOr(&cb[idx_start_scb+j], shared_cb[j]);
	}
#endif
}

__global__ void bccb_kernel_hybrid_tmp_buf(unsigned int* codes, size_t n, unsigned int** tmp_cbs, unsigned int* cb)
{
	__shared__ int shared_cb[NB_INT_SHARED];

	// First stage is to set shared_cb to 0
	for (int i = threadIdx.x; i < NB_INT_SHARED; i += blockDim.x) {
		shared_cb[i] = 0;
	}

	__syncthreads();

	int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;
	int size_grid = blockDim.x*gridDim.x*gridDim.y;

	// blockIdx.y gives the pass that we must care about
	int idx_start_scb = blockIdx.y*NB_INT_SHARED;
	int idx_end_scb = idx_start_scb + NB_INT_SHARED;

	unsigned int* tmp_cb = tmp_cbs[blockIdx.y];

	for (; idx < n; idx += size_grid) {
		int p1 = codes[idx];
		const int idx_scb1 = p1>>5;
		const unsigned int rem1 = ((unsigned int)1)<<(p1&31);

		if ((idx_scb1 >= idx_start_scb) && (idx_scb1 < idx_end_scb)) {
			atomicOr(&shared_cb[idx_scb1 - idx_start_scb], rem1);
		}
		else {
			atomicOr(&tmp_cb[idx_scb1], rem1);
		}
	}

#ifdef LAST_PASS_DIFFERENT
	int nb_int_shared;
	if (blockIdx.y == gridDim.y-1) {
		// Last pass has less integer into the global buffer
		nb_int_shared = NB_INT_SHARED_LAST_PASS;
	}
	else {
		nb_int_shared = NB_INT_SHARED;
	}
	__syncthreads();

	// Merge shared_cb into the global cb
	for (int j = threadIdx.x; j < nb_int_shared; j += blockDim.x) {
		atomicOr(&tmp_cb[idx_start_scb+j], shared_cb[j]);
	}
#else
	__syncthreads();

	// Merge shared_cb into the global cb
	for (int j = threadIdx.x; j < NB_INT_SHARED; j += blockDim.x) {
		atomicOr(&tmp_cb[idx_start_scb+j], shared_cb[j]);
	}
#endif
}

__global__ void bccb_kernel_only_shared(unsigned int* codes, size_t n, unsigned int* cb)
{
	__shared__ int shared_cb[NB_INT_SHARED];

	// First stage is to set shared_cb to 0
	for (int i = threadIdx.x; i < NB_INT_SHARED; i += blockDim.x) {
		shared_cb[i] = 0;
	}

	__syncthreads();

	// Second stage is to compute the values of this block into shared_cb

	// Comptue the index of the point being process
	int size_grid = blockDim.x*gridDim.x;

	int idx_start_scb, idx_end_scb;
	for (int i = 0; i < NB_PASSES-1; i++) {
		idx_start_scb = i*NB_INT_SHARED;
		idx_end_scb = idx_start_scb + NB_INT_SHARED;
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		while (idx < n) {
			// Get the point
			int b = codes[idx];
			idx += size_grid;
			const int idx_scb = b>>5;
			if ((idx_scb >= idx_start_scb) && (idx_scb < idx_end_scb)) {
				atomicOr(&shared_cb[idx_scb-idx_start_scb], ((unsigned int)1)<<(b&31));
			}
		}

		__syncthreads();
	}
}

__global__ void bccb_kernel_one_pass(unsigned int* codes, size_t n, unsigned int* cb)
{
	__shared__ int shared_cb[NB_INT_SHARED];

	// First stage is to set shared_cb to 0
	for (int i = threadIdx.x; i < NB_INT_SHARED; i += blockDim.x) {
		shared_cb[i] = 0;
	}

	__syncthreads();

	// Second stage is to compute the values of this block into shared_cb

	// Comptue the index of the point being process
	int size_grid = blockDim.x*gridDim.x;

	// blockIdx.y gives the pass that we must care about
	int idx_start_scb = blockIdx.y*NB_INT_SHARED;
	int idx_end_scb = idx_start_scb + NB_INT_SHARED;

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while (idx < n) {
		// Get the point
		int b = codes[idx];
		idx += size_grid;
		const int idx_scb = b>>5;
		if ((idx_scb >= idx_start_scb) && (idx_scb < idx_end_scb)) {
			atomicOr(&shared_cb[idx_scb-idx_start_scb], ((unsigned int)1)<<(b&31));
		}
	}

#ifdef LAST_PASS_DIFFERENT
	int nb_int_shared;
	if (blockIdx.y == gridDim.y-1) {
		// Last pass has less integer into the global buffer
		nb_int_shared = NB_INT_SHARED_LAST_PASS;
	}
	else {
		nb_int_shared = NB_INT_SHARED;
	}
	__syncthreads();

	// Merge shared_cb into the global cb
	for (int j = threadIdx.x; j < nb_int_shared; j += blockDim.x) {
		atomicOr(&cb[idx_start_scb+j], shared_cb[j]);
		shared_cb[j] = 0;
	}
#else
	__syncthreads();

	// Merge shared_cb into the global cb
	for (int j = threadIdx.x; j < NB_INT_SHARED; j += blockDim.x) {
		atomicOr(&cb[idx_start_scb+j], shared_cb[j]);
		shared_cb[j] = 0;
	}
#endif
}

__global__ void bccb_kernel_n_pass(unsigned int* codes, size_t n, unsigned int* cb, int npasses)
{
	__shared__ int shared_cb[NB_INT_SHARED];

	// First stage is to set shared_cb to 0
	for (int i = threadIdx.x; i < NB_INT_CB_SHARED; i += blockDim.x) {
		shared_cb[i] = 0;
	}

	__syncthreads();

	// Second stage is to compute the values of this block into shared_cb

	// Comptue the index of the point being process
	int size_grid = blockDim.x*gridDim.x;

	// blockIdx.y gives the pass that we must care about
	int idx_pass = blockIdx.y;

	while (idx_pass < NB_PASSES) {
		int idx_start_scb = idx_pass*NB_INT_SHARED;
		int idx_end_scb = idx_start_scb + NB_INT_SHARED;
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		while (idx < n) {
			// Get the point
			int b = codes[idx];
			idx += size_grid;
			const int idx_scb = b>>5;
			if ((idx_scb >= idx_start_scb) && (idx_scb < idx_end_scb)) {
				atomicOr(&shared_cb[idx_scb-idx_start_scb], ((unsigned int)1)<<(b&31));
			}
		}

#ifdef LAST_PASS_DIFFERENT
		int nb_int_shared;
		if (blockIdx.y == gridDim.y-1) {
			// Last pass has less integer into the global buffer
			nb_int_shared = NB_INT_SHARED_LAST_PASS;
		}
		else {
			nb_int_shared = NB_INT_SHARED;
		}
		__syncthreads();

		// Merge shared_cb into the global cb
		for (int j = threadIdx.x; j < nb_int_shared; j += blockDim.x) {
			atomicOr(&cb[idx_start_scb+j], shared_cb[j]);
			shared_cb[j] = 0;
		}
#else
		__syncthreads();

		// Merge shared_cb into the global cb
		for (int j = threadIdx.x; j < NB_INT_SHARED; j += blockDim.x) {
			atomicOr(&cb[idx_start_scb+j], shared_cb[j]);
			shared_cb[j] = 0;
		}
#endif
		idx_pass += gridDim.y;
	}
}

__global__ void bccb_kernel_n_pass_cache(unsigned int* codes, size_t n, unsigned int* cb, int npasses)
{
	__shared__ int shared_cb[NB_INT_SHARED];

	// First stage is to set shared_cb to 0
	for (int i = threadIdx.x; i < NB_INT_CB_SHARED; i += blockDim.x) {
		shared_cb[i] = 0;
	}

	__syncthreads();

	// Second stage is to compute the values of this block into shared_cb

	// Comptue the index of the point being process
	int size_grid = blockDim.x*gridDim.x;

	// blockIdx.y gives the pass that we must care about
	int idx_pass = blockIdx.y;

	while (idx_pass < NB_PASSES) {
		int idx_start_scb = idx_pass*NB_INT_SHARED;
		int idx_end_scb = idx_start_scb + NB_INT_SHARED;
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		while (idx < n) {
			// Get the point
			int b = codes[idx];
			idx += size_grid;
			const int idx_scb = b>>5;
			if ((idx_scb >= idx_start_scb) && (idx_scb < idx_end_scb)) {
				atomicOr(&shared_cb[idx_scb-idx_start_scb], ((unsigned int)1)<<(b&31));
			}
		}

#ifdef LAST_PASS_DIFFERENT
		int nb_int_shared;
		if (blockIdx.y == gridDim.y-1) {
			// Last pass has less integer into the global buffer
			nb_int_shared = NB_INT_SHARED_LAST_PASS;
		}
		else {
			nb_int_shared = NB_INT_SHARED;
		}
		__syncthreads();

		// Merge shared_cb into the global cb
		for (int j = threadIdx.x; j < nb_int_shared; j += blockDim.x) {
			atomicOr(&cb[idx_start_scb+j], shared_cb[j]);
			shared_cb[j] = 0;
		}
#else
		__syncthreads();

		// Merge shared_cb into the global cb
		for (int j = threadIdx.x; j < NB_INT_SHARED; j += blockDim.x) {
			atomicOr(&cb[idx_start_scb+j], shared_cb[j]);
			shared_cb[j] = 0;
		}
#endif
		idx_pass += gridDim.y;
	}
}

__global__ void bccb_kernel(unsigned int* codes, size_t n, unsigned int* cb)
{
	__shared__ int shared_cb[NB_INT_SHARED];

	// First stage is to set shared_cb to 0
	for (int i = threadIdx.x; i < NB_INT_SHARED; i += blockDim.x) {
		shared_cb[i] = 0;
	}

	__syncthreads();

	// Second stage is to compute the values of this block into shared_cb

	// Comptue the index of the point being process
	int size_grid = blockDim.x*gridDim.x;

	int idx_start_scb, idx_end_scb;
#pragma unroll 20
	for (int i = 0; i < NB_PASSES-1; i++) {
		idx_start_scb = i*NB_INT_SHARED;
		idx_end_scb = idx_start_scb + NB_INT_SHARED;
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		while (idx < n) {
			// Get the point
			int b = codes[idx];
			idx += size_grid;
			const int idx_scb = b>>5;
			if ((idx_scb >= idx_start_scb) && (idx_scb < idx_end_scb)) {
				atomicOr(&shared_cb[idx_scb-idx_start_scb], ((unsigned int)1)<<(b&31));
			}
		}

		__syncthreads();

		// Merge shared_cb into the global cb
		for (int j = threadIdx.x; j < NB_INT_SHARED; j += blockDim.x) {
			atomicOr(&cb[idx_start_scb+j], shared_cb[j]);
			shared_cb[j] = 0;
		}

		__syncthreads();
	}

	__syncthreads();

	idx_start_scb = (NB_PASSES-1)*NB_INT_SHARED;
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while (idx < n) {
		// Get the point
		int b = codes[idx];
		idx += size_grid;
		const int idx_scb = b>>5;
		if (idx_scb >= idx_start_scb) {
			atomicOr(&shared_cb[idx_scb-idx_start_scb], ((unsigned int)1)<<(b&31));
		}
	}

	__syncthreads();

	// Merge shared_cb into the global cb
	for (int i = threadIdx.x; i < NB_INT_SHARED_LAST_PASS; i += blockDim.x) {
		atomicOr(&cb[idx_start_scb+i], shared_cb[i]);
	}
}

#define NPASS_CACHE 2
__global__ void bccb_kernel_cache(unsigned int* codes, size_t n, unsigned int* cb, int npass_cache)
{
	__shared__ int shared_cb[NB_INT_SHARED];

	// First stage is to set shared_cb to 0
	for (int i = threadIdx.x; i < NB_INT_SHARED; i += blockDim.x) {
		shared_cb[i] = 0;
	}

	__syncthreads();

	// Second stage is to compute the values of this block into shared_cb
	int size_grid = blockDim.x*gridDim.x;

	// When two "size_grid" round of codes have been loaded, then it fits into L2 cache
	// We should process all the passes will these data are in cache.
	int idx_start_scb, idx_end_scb;
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
#pragma unroll 4
	while (idx < n) {
		for (int i = 0; i < NB_PASSES-1; i++) {
			idx_start_scb = i*NB_INT_SHARED;
			idx_end_scb = idx_start_scb + NB_INT_SHARED;
#pragma unroll 2
			for (int k = 0; k < npass_cache; k++) {
				// Get the point
				int b = codes[idx+k*size_grid];
				const int idx_scb = b>>5;
				if ((idx_scb >= idx_start_scb) && (idx_scb < idx_end_scb)) {
					atomicOr(&shared_cb[idx_scb-idx_start_scb], ((unsigned int)1)<<(b&31));
				}
				if (idx >= n) {
					break;
				}
			}

			__syncthreads();

		}
		idx += npass_cache*size_grid;
	}

	// Merge shared_cb into the global cb
	for (int j = threadIdx.x; j < NB_INT_SHARED; j += blockDim.x) {
		atomicOr(&cb[idx_start_scb+j], shared_cb[j]);
		shared_cb[j] = 0;
	}

	__syncthreads();
/*
	idx_start_scb = (NB_PASSES-1)*NB_INT_SHARED;
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	//while (idx < n) {
	for (int k = 0; k < n_k; k++) {
		// Get the point
		int b = codes[idx];
		idx += size_grid;
		const int idx_scb = b>>5;
		if (idx_scb >= idx_start_scb) {
			atomicOr(&shared_cb[idx_scb-idx_start_scb], ((unsigned int)1)<<(b&31));
		}
		k++;
	}

	__syncthreads();

	// Merge shared_cb into the global cb
	for (int i = threadIdx.x; i < NB_INT_SHARED_LAST_PASS; i += blockDim.x) {
		atomicOr(&cb[idx_start_scb+i], shared_cb[i]);
	}*/
}

void gpu_bccb_init()
{
}

void gpu_bccb(PVBCode* codes, size_t n, BCodeCB cb)
{
	PVBCode* device_codes;
	BCodeCB device_cb;

	verify_cuda(cudaMalloc(&device_codes, n*sizeof(PVBCode)));
	verify_cuda(cudaMalloc(&device_cb, SIZE_BCODECB));

	verify_cuda(cudaMemset(device_cb, 0x00, SIZE_BCODECB));
	verify_cuda(cudaMemcpy(device_codes, codes, n*sizeof(PVBCode), cudaMemcpyHostToDevice));

	cudaEvent_t start,end;
	verify_cuda(cudaEventCreate(&start));
	verify_cuda(cudaEventCreate(&end));

	//int nblocks = get_number_blocks()*2;
	int nblocks = 28;
	int size_grid = nblocks*NTHREADS_BLOCK;
	/*verify(n % 8*size_grid == 0);
	verify(n > 8*size_grid);*/
	fprintf(stderr, "Grid size is %d.\n", size_grid);
	fprintf(stderr, "Nb int bcodecb: %d/%d | Nb int shared cb: %d | Nb passes: %d\n", NB_INT_BCODECB, SIZE_BCODECB, NB_INT_SHARED, NB_PASSES);

	verify_cuda(cudaEventRecord(start, 0));
	bccb_kernel<<<dim3(nblocks,1),NTHREADS_BLOCK>>>((unsigned int*) device_codes, n, device_cb);
	//bccb_kernel_cache<<<nblocks,NTHREADS_BLOCK>>>((unsigned int*) device_codes, n, device_cb, 7);
	verify_cuda_kernel();
	verify_cuda(cudaEventRecord(end, 0));
	verify_cuda(cudaEventSynchronize(end));

	verify_cuda(cudaMemcpy(cb, device_cb, SIZE_BCODECB, cudaMemcpyDeviceToHost));

	verify_cuda(cudaFree(device_codes));
	verify_cuda(cudaFree(device_cb));

	float time = 0;
	verify_cuda(cudaEventElapsedTime(&time, start, end));

	fprintf(stderr, "CUDA kernel time (%d blocks): %0.4f ms, BW: %0.4f MB/s\n", nblocks, time, ((float)(n*sizeof(PVBCode)))/(1024.0f*1024.0f*(time/1000.0f)));
}

void gpu_bccb_2dim(PVBCode* codes, size_t n, BCodeCB cb)
{
	PVBCode* device_codes;
	BCodeCB device_cb;

	verify_cuda(cudaMalloc(&device_codes, n*sizeof(PVBCode)));
	verify_cuda(cudaMalloc(&device_cb, SIZE_BCODECB));

	verify_cuda(cudaMemset(device_cb, 0x00, SIZE_BCODECB));
	verify_cuda(cudaMemcpy(device_codes, codes, n*sizeof(PVBCode), cudaMemcpyHostToDevice));

	unsigned int* tmp_cbs[NB_PASSES];
	for (int i = 0; i < NB_PASSES; i++) {
		verify_cuda(cudaMalloc(&tmp_cbs[i], SIZE_BCODECB));
		verify_cuda(cudaMemset(tmp_cbs[i], 0x00, SIZE_BCODECB));
	}
	unsigned int** dev_tmp_cbs;
	cudaMalloc(&dev_tmp_cbs, sizeof(unsigned int*)*NB_PASSES);
	cudaMemcpy(dev_tmp_cbs, tmp_cbs, sizeof(unsigned int*)*NB_PASSES, cudaMemcpyHostToDevice);

	cudaEvent_t start,end;
	verify_cuda(cudaEventCreate(&start));
	verify_cuda(cudaEventCreate(&end));

	int nblocks_x = 14;
	int nblocks_y = NB_PASSES;
	int size_grid = nblocks_x*NTHREADS_BLOCK*nblocks_y;
	fprintf(stderr, "Grid size is %d.\n", size_grid);
	fprintf(stderr, "Nb int bcodecb: %d/%d | Nb int shared cb: %d | Nb int last : %d | Nb passes: %d\n", NB_INT_BCODECB, SIZE_BCODECB, NB_INT_SHARED, NB_INT_SHARED_LAST_PASS, NB_PASSES);

	dim3 nblocks(nblocks_x, nblocks_y);
	verify_cuda(cudaEventRecord(start, 0));
	bccb_kernel_one_pass<<<nblocks,NTHREADS_BLOCK>>>((unsigned int*) device_codes, n, device_cb);
	//bccb_kernel_hybrid<<<nblocks,NTHREADS_BLOCK>>>((unsigned int*) device_codes, n, device_cb);
	//bccb_kernel_hybrid_tmp_buf<<<nblocks,NTHREADS_BLOCK>>>((unsigned int*) device_codes, n, dev_tmp_cbs, device_cb);
	verify_cuda_kernel();
	verify_cuda(cudaEventRecord(end, 0));
	verify_cuda(cudaEventSynchronize(end));

	verify_cuda(cudaMemcpy(cb, device_cb, SIZE_BCODECB, cudaMemcpyDeviceToHost));

	verify_cuda(cudaFree(device_codes));
	verify_cuda(cudaFree(device_cb));

	for (int i = 0; i < NB_PASSES; i++) {
		verify_cuda(cudaFree(tmp_cbs[i]));
	}

	float time = 0;
	verify_cuda(cudaEventElapsedTime(&time, start, end));

	fprintf(stderr, "CUDA kernel time (%d x %d blocks): %0.4f ms, BW: %0.4f MB/s\n", nblocks_x, nblocks_y, time, ((float)(n*sizeof(PVBCode)))/(1024.0f*1024.0f*(time/1000.0f)));
}

void gpu_bccb_launch_kernel(PVBCode* dev_codes, size_t n, BCodeCB device_cb, int nblocks, cudaEvent_t event, cudaStream_t stream)
{
	bccb_kernel<<<nblocks,NTHREADS_BLOCK,0,stream>>>((unsigned int*) dev_codes, n, device_cb);
	if (event) {
		cudaEventRecord(event, 0);
	}
	verify_cuda_kernel();
}
