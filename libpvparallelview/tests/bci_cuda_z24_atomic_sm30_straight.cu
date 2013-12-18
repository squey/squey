/**
 * \file bci_cuda_z24.cu
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVHSVColor.h>

#include <pvkernel/cuda/common.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCICode.h>
#include "bci_cuda_30.h"

#include <cassert>
#include <iostream>

#define NTHREADS_BLOCK 512
#define SMEM_IMG_KB (4*4)

// From http://code.google.com/p/cudaraster/source/browse/trunk/src/cudaraster/cuda/Util.hpp?r=4
// See ptx_isa_3.0.pdf in CUDA SDK documentation for more information on prmt.b32
__device__ __inline__ unsigned int prmt(unsigned int a, unsigned int b, unsigned int c)
{
	unsigned int v;
	asm("prmt.b32 %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c));
	return v;
}

using PVParallelView::PVBCICode;

//#define MASK_ZBUFFER 0x00FFFFFF
#define MASK_ZBUFFER 0xFFFFFF00
#define MASK_COLOR   0x000000FF
#define BCI_MASK_TYPE ((1<<2)-1)

#pragma pack(push)
#pragma pack(4)
struct img_zbuffer_t
{
	union {
		uint32_t int_v;
		struct {
			uint8_t zbuffer[3];
			uint8_t hsv;
		} s;
	};
};
#pragma pack(pop)

__device__ __inline__ unsigned char zone2pos(unsigned char zone)
{
	const unsigned char a0 = zone&1;
	const unsigned char a1 = (zone&2)>>1;
	const unsigned char a2 = zone&4;

	return ((!(a0 ^ a1)) & (!a2)) |
	      (((a1 & (!a0)) | ((a2>>2) & a0)) << 1);
}

__device__ __inline__ unsigned char plus1mod3(unsigned char i)
{
	//return (i+1)%3;
	const unsigned char a0 = i&1;
	const unsigned char a1 = i&2;

	return (!i) | (((!a1) & a0)<<1);

}

__device__ __noinline__ unsigned int hsv2rgb(unsigned int hsv)
{
	// We have:
	// hsv defines actually only h which is divided in zone of 2**HSV_COLOR_NBITS_ZONE numbers.
	// Thus, we need to compute the zone, pos and mask, and then
	// v = {R, G, B}
	// v[pos] = x ^ mask
	// v[(pos+1)%3] = 0 ^ mask
	// v[(pos+2)%3] = 255 ^ mask
	
	if (hsv == HSV_COLOR_WHITE) {
		return 0xFFFFFFFF; // Special value for white
	}
	if (hsv == HSV_COLOR_BLACK) {
		return 0xFF000000; // Special value for black
	}

	unsigned char zone = (unsigned char) (hsv>>HSV_COLOR_NBITS_ZONE);
	unsigned char pos = zone2pos(zone);
	unsigned char mask = (zone&1)*0xFF;
	
	unsigned int pre_perm0, pre_perm1, pre_perm2,pre_perm,perm;
	pre_perm0 = (((hsv&HSV_COLOR_MASK_ZONE)*255)>>HSV_COLOR_NBITS_ZONE) ^ mask;
	pre_perm1 = mask;
	pre_perm2 = 0xFF ^ mask;
	asm("mov.u32 %0,{%1,%2,%3,%4};" : "=r"(pre_perm) : "r"(pre_perm0), "r"(pre_perm1), "r"(pre_perm2), "r"((unsigned int)0xFF));

	const unsigned int pos2 = plus1mod3(pos);
	perm = (1 << (pos2<<2)) | (2 << ((plus1mod3(pos2))<<2)) | (3 << 12);

	return prmt(pre_perm, pre_perm, perm);
}

template <size_t Bbits, bool width_reverse>
__global__ void bcicode_raster_unroll2(uint2* bci_codes, unsigned int n, unsigned int width,  unsigned int* img_dst, unsigned int img_width, unsigned int img_x_start, const float zoom_y)
{
	// shared_size = blockDim.x*IMAGE_HEIGHT*sizeof(img_zbuffer_t)
	__shared__ unsigned int shared_img[(SMEM_IMG_KB*1024)/sizeof(unsigned int)];

	// The x coordinate of the band this thread is responsible of
	int band_x = threadIdx.x + blockIdx.x*blockDim.x;
	if (band_x >= width) {
		return;
	}

	// Do this division once and for all
	const float alpha0 = (float)(width-band_x)/(float)width;
	const float alpha1 = (float)(width-(band_x+1))/(float)width;
	const unsigned int y_start = threadIdx.y + blockIdx.y*blockDim.y;
	const unsigned int size_grid = blockDim.y*gridDim.y;

	// First stage is to set shared memory
	for (int y = threadIdx.y; y < PVParallelView::constants<Bbits>::image_height; y += blockDim.y) {
		shared_img[threadIdx.x + y*blockDim.x] = 0xFFFFFFFF;
	}

	__syncthreads();

	unsigned int idx_codes = y_start;
	for (; idx_codes < n; idx_codes += size_grid) {
		uint2 code0 = bci_codes[idx_codes];
		code0.x &= MASK_ZBUFFER;
		const float l0 = (float) (code0.y & PVParallelView::constants<Bbits>::mask_int_ycoord);
		const int r0i = (code0.y >> Bbits) & PVParallelView::constants<Bbits>::mask_int_ycoord;
		int pixel_y00;
		int pixel_y01;
		const float r0 = (float) r0i;
		pixel_y00 = (int) (((r0 + ((l0-r0)*alpha0)) * zoom_y) + 0.5f);
		pixel_y01 = (int) (((r0 + ((l0-r0)*alpha1)) * zoom_y) + 0.5f);

		if (pixel_y00 > pixel_y01) {
			const int tmp = pixel_y00;
			pixel_y00 = pixel_y01;
			pixel_y01 = tmp;
		}

		const unsigned int color0 = (code0.y >> 2*Bbits) & 0xff;
		if (color0 == HSV_COLOR_BLACK) { // Used for zombie events, so their index is the highest possible (behind everyone)
			code0.x = MASK_ZBUFFER;
		}
		const unsigned int shared_v = color0 | code0.x;
		atomicMin(&shared_img[threadIdx.x + pixel_y00*blockDim.x], shared_v);
		for (int pixel_y0 = pixel_y00+1; pixel_y0 < pixel_y01; pixel_y0++) {
			atomicMin(&shared_img[threadIdx.x + pixel_y0*blockDim.x], shared_v);
		}
	}
	
	band_x += img_x_start;
	if (width_reverse) {
		band_x = img_width-band_x-1;
	}
	__syncthreads();
	// Final stage is to commit the shared image into the global image
	for (int y = threadIdx.y; y < PVParallelView::constants<Bbits>::image_height; y += blockDim.y) {
		const unsigned int pixel_shared = shared_img[threadIdx.x + y*blockDim.x];
		unsigned int pixel;
		if (pixel_shared != 0xFFFFFFFF) {
			pixel = hsv2rgb(pixel_shared & MASK_COLOR);
		}
		else {
			pixel = 0x00000000; // Transparent background
		}
		img_dst[band_x + y*img_width] = pixel;
	}
}

template <size_t Bbits>
static inline int get_nthread_x_from_width(int width)
{
	return (picviz_min(width, (SMEM_IMG_KB*1024)/(PVParallelView::constants<Bbits>::image_height*sizeof(img_zbuffer_t))));
}

template <size_t Bbits, bool reverse>
static void show_codes_cuda(PVParallelView::PVBCICode<Bbits>* device_codes, uint32_t n, uint32_t width, uint32_t* device_img, uint32_t img_width, uint32_t x_start, const float zoom_y, cudaStream_t stream)
{
	if (zoom_y == 0) {
		return;
	}
	assert((zoom_y > 0) && (zoom_y <= 1.0f));
	// Compute number of threads per block
	int nthreads_x = get_nthread_x_from_width<Bbits>(width);
	int nthreads_y = NTHREADS_BLOCK/nthreads_x;
	assert(nthreads_x*nthreads_y <= NTHREADS_BLOCK);

	// Compute number of blocks
	int nblocks = PVCuda::get_number_blocks();
	int nblocks_x = (width+nthreads_x-1)/nthreads_x;
	int nblocks_y = 1;

	// Launch CUDA kernel!
	bcicode_raster_unroll2<Bbits, reverse><<<dim3(nblocks_x,nblocks_y),dim3(nthreads_x, nthreads_y), 0, stream>>>((uint2*) device_codes, n, width, device_img, img_width, x_start, zoom_y);
}

template <size_t Bbits, bool reverse>
static float show_and_perf_codes_cuda(PVParallelView::PVBCICode<Bbits>* device_codes, uint32_t n, uint32_t width, uint32_t* device_img, uint32_t img_width, uint32_t x_start, const float zoom_y, cudaStream_t stream, double* kernel_bw)
{
	// WARNING!
	// This will imply a synchronous call to the CUDA kernel, thus must not be used
	// inside the Picviz rendering pipeline!!
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreateWithFlags(&stop, cudaEventBlockingSync);

	cudaEventRecord(start, stream);
	show_codes_cuda<Bbits, reverse>(device_codes, n, width, device_img, img_width, x_start, zoom_y, stream);
	picviz_verify_cuda_kernel();
	cudaEventRecord(stop, stream);
	cudaEventSynchronize(stop);

	float time = 0.0f;
	cudaEventElapsedTime(&time, start, stop);
	if (kernel_bw) {
		*kernel_bw = (double)(n*sizeof(PVBCICode<Bbits>))/(double)((time/1000.0)*1024.0*1024.0);
	}

	return time;
}

float show_and_perf_codes_cuda10_test(PVParallelView::PVBCICode<10>* device_codes, uint32_t n, uint32_t width, uint32_t* device_img, uint32_t img_width, uint32_t x_start, const float zoom_y, cudaStream_t stream, double* bw)
{
	return show_and_perf_codes_cuda<10, false>(device_codes, n, width, device_img, img_width, x_start, zoom_y, stream, bw);
}

float show_and_perf_codes_cuda11_test(PVParallelView::PVBCICode<11>* device_codes, uint32_t n, uint32_t width, uint32_t* device_img, uint32_t img_width, uint32_t x_start, const float zoom_y, cudaStream_t stream, double* bw)
{
	return show_and_perf_codes_cuda<11, false>(device_codes, n, width, device_img, img_width, x_start, zoom_y, stream, bw);
}
