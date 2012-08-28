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
#include <pvparallelview/cuda/bci_cuda.h>

#include <cassert>

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

	const unsigned int size_grid2 = size_grid<<1;
	const unsigned int n_end = (n/(size_grid2))*(size_grid2);

	__syncthreads();

	unsigned int idx_codes = y_start;
#if 0
	for (; idx_codes < n_end; idx_codes += size_grid2) {
		uint2 code0 = bci_codes[idx_codes];
		uint2 code1 = bci_codes[idx_codes+size_grid];
		uint2 code2 = bci_codes[idx_codes+size_grid*2];
		uint2 code3 = bci_codes[idx_codes+size_grid*3];

		/*if (threadIdx.x == 0 && threadIdx.y == 0) {
			i++;
		}*/
		
		// For information:
		// struct PVBCICode
		// {
		//	typedef PVCore::PVAlignedAllocator<PVBCICode, 16> allocator;
		//	union {
		//		uint64_t int_v;
		//		struct {
		//			uint32_t idx;
		//			uint32_t l: 10;
		//			uint32_t r: 10;
		//			uint32_t color: 9;
		//			uint32_t __reserved: 1;
		//		} s;
		//	};
		// }

		// 24-bit z-buffer
		code0.x >>= 8; code1.x >>= 8; code2.x >>= 8; code3.x >>= 8;

		// Get l, r and color
		const float l0 = (float) (code0.y & 0x3ff);
		const float r0 = (float) ((code0.y & 0xffc00)>>10);
		const float l1 = (float) (code1.y & 0x3ff);
		const float r1 = (float) ((code1.y & 0xffc00)>>10);
		const float l2 = (float) (code2.y & 0x3ff);
		const float r2 = (float) ((code2.y & 0xffc00)>>10);
		const float l3 = (float) (code3.y & 0x3ff);
		const float r3 = (float) ((code3.y & 0xffc00)>>10);

		// Compute the y coordinate for band_x
		const int pixel_y0 = (int) (r0 + ((l0-r0)*alpha) + 0.5f);
		const int pixel_y1 = (int) (r1 + ((l1-r1)*alpha) + 0.5f);
		const int pixel_y2 = (int) (r2 + ((l2-r2)*alpha) + 0.5f);
		const int pixel_y3 = (int) (r3 + ((l3-r3)*alpha) + 0.5f);
		unsigned int idx_shared_img0 = threadIdx.x + pixel_y0*blockDim.x;
		unsigned int idx_shared_img1 = threadIdx.x + pixel_y1*blockDim.x;
		unsigned int idx_shared_img2 = threadIdx.x + pixel_y2*blockDim.x;
		unsigned int idx_shared_img3 = threadIdx.x + pixel_y3*blockDim.x;
		const unsigned int color0 = (code0.y & 0xff00000)<<4;
		const unsigned int color1 = (code1.y & 0xff00000)<<4;
		const unsigned int color2 = (code2.y & 0xff00000)<<4;
		const unsigned int color3 = (code3.y & 0xff00000)<<4;

		// Set shared_img
		unsigned int cur_shared_p0 = shared_img[idx_shared_img0] & MASK_ZBUFFER;
		if (cur_shared_p0 > code0.x) {
			shared_img[idx_shared_img0] = color0 | code0.x;
		}
		unsigned int cur_shared_p1 = shared_img[idx_shared_img1] & MASK_ZBUFFER;
		if (cur_shared_p1 > code1.x) {
			shared_img[idx_shared_img1] = color1 | code1.x;
		}
		unsigned int cur_shared_p2 = shared_img[idx_shared_img2] & MASK_ZBUFFER;
		if (cur_shared_p2 > code2.x) {
			shared_img[idx_shared_img2] = color2 | code2.x;
		}
		unsigned int cur_shared_p3 = shared_img[idx_shared_img3] & MASK_ZBUFFER;
		if (cur_shared_p3 > code3.x) {
			shared_img[idx_shared_img3] = color3 | code3.x;
		}
		__syncthreads();
	}
	for (; idx_codes < n_end; idx_codes += size_grid2) {
		uint2 code0 = bci_codes[idx_codes];
		uint2 code1 = bci_codes[idx_codes+size_grid];
		
		// For information:
		// struct PVBCICode
		// {
		//	typedef PVCore::PVAlignedAllocator<PVBCICode, 16> allocator;
		//	union {
		//		uint64_t int_v;
		//		struct {
		//			uint32_t idx;
		//			uint32_t l: 10;
		//			uint32_t r: 10;
		//			uint32_t color: 9;
		//			uint32_t __reserved: 1;
		//		} s;
		//	};
		// }

		// 24-bit z-buffer
		code0.x &= MASK_ZBUFFER; code1.x &= MASK_ZBUFFER;

		// Get l, r and color
		const float l0 = (float) (code0.y & PVParallelView::constants<Bbits>::mask_int_ycoord);
		const float r0 = (float) ((code0.y >> Bbits) & PVParallelView::constants<Bbits>::mask_int_ycoord);
		const float l1 = (float) (code1.y & PVParallelView::constants<Bbits>::mask_int_ycoord);
		const float r1 = (float) ((code1.y >> Bbits) & PVParallelView::constants<Bbits>::mask_int_ycoord);

		// Compute the y coordinate for band_x
		int pixel_y00 = (int) (((r0 + ((l0-r0)*alpha0)) * zoom_y) + 0.5f);
		int pixel_y01 = (int) (((r0 + ((l0-r0)*alpha1)) * zoom_y) + 0.5f);
		if (pixel_y00 > pixel_y01) {
			const int tmp = pixel_y00;
			pixel_y00 = pixel_y01;
			pixel_y01 = tmp;
		}
		int pixel_y10 = (int) (((r1 + ((l1-r1)*alpha0)) * zoom_y) + 0.5f);
		int pixel_y11 = (int) (((r1 + ((l1-r1)*alpha1)) * zoom_y) + 0.5f);
		if (pixel_y10 > pixel_y11) {
			const int tmp = pixel_y10;
			pixel_y10 = pixel_y11;
			pixel_y11 = tmp;
		}
		const unsigned int shared_v0 = ((code0.y >> 2*Bbits) & 0xff) | code0.x;
		const unsigned int shared_v1 = ((code1.y >> 2*Bbits) & 0xff) | code1.x;

		atomicMin(&shared_img[threadIdx.x + pixel_y00*blockDim.x], shared_v0);
		atomicMin(&shared_img[threadIdx.x + pixel_y10*blockDim.x], shared_v1);

		for (int pixel_y0 = pixel_y00+1; pixel_y0 < pixel_y01; pixel_y0++) {
			atomicMin(&shared_img[threadIdx.x + pixel_y0*blockDim.x], shared_v0);
		}
		for (int pixel_y1 = pixel_y10+1; pixel_y1 < pixel_y11; pixel_y1++) {
			atomicMin(&shared_img[threadIdx.x + pixel_y1*blockDim.x], shared_v1);
		}
	}
#endif
	for (; idx_codes < n; idx_codes += size_grid) {
		uint2 code0 = bci_codes[idx_codes];
		code0.x &= MASK_ZBUFFER;
		const float l0 = (float) (code0.y & PVParallelView::constants<Bbits>::mask_int_ycoord);
		const int r0i = (code0.y >> Bbits) & PVParallelView::constants<Bbits>::mask_int_ycoord;
		const int type = (code0.y >> ((2*Bbits) + 8)) & BCI_MASK_TYPE;
		int pixel_y00;
		int pixel_y01;
		if (type == PVBCICode<Bbits>::STRAIGHT) {
			const float r0 = (float) r0i;
			pixel_y00 = (int) (((r0 + ((l0-r0)*alpha0)) * zoom_y) + 0.5f);
			pixel_y01 = (int) (((r0 + ((l0-r0)*alpha1)) * zoom_y) + 0.5f);
		}
		else {
			// TODO: optimise this !!
			// 'r0i' is x
			if (band_x > r0i) {
				// This is out of our drawing scope !
				continue;
			}
			const float r0 = (float) r0i;
			if (type == PVBCICode<Bbits>::UP) {
				const float alpha_x = l0/r0;
				pixel_y00 = (int) (((l0-(alpha_x*(float)band_x))*zoom_y) + 0.5f);
				if (band_x == r0i) {
					pixel_y01 = pixel_y00;
				}
				else {
					pixel_y01 = (int) (((l0-(alpha_x*(float)(band_x+1)))*zoom_y) + 0.5f);
				}
			}
			else {
				const float alpha_x = ((float)PVParallelView::constants<Bbits>::mask_int_ycoord-l0)/r0;
				pixel_y00 = (int) (((l0+(alpha_x*(float)band_x))*zoom_y) + 0.5f);
				if (band_x == r0i) {
					pixel_y01 = pixel_y00;
				}
				else {
					pixel_y01 = (int) (((l0+(alpha_x*(float)(band_x+1)))*zoom_y) + 0.5f);
				}
			}
		}

		if (pixel_y00 > pixel_y01) {
			const int tmp = pixel_y00;
			pixel_y00 = pixel_y01;
			pixel_y01 = tmp;
		}

		const unsigned int color0 = (code0.y >> 2*Bbits) & 0xff;
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

template <size_t Bbits, bool reverse>
static void show_codes_cuda(PVParallelView::PVBCICode<Bbits>* device_codes, uint32_t n, uint32_t width, uint32_t* device_img, uint32_t img_width, uint32_t x_start, const float zoom_y, cudaStream_t stream)
{
	assert((zoom_y > 0) && (zoom_y <= 1.0f));
	// Compute number of threads per block
	int nthreads_x = (picviz_min(width, (SMEM_IMG_KB*1024)/(PVParallelView::constants<Bbits>::image_height*sizeof(img_zbuffer_t))));
	int nthreads_y = NTHREADS_BLOCK/nthreads_x;
	assert(nthreads_x*nthreads_y <= NTHREADS_BLOCK);

	PVLOG_INFO("Number threads per block: %d x %d\n", nthreads_x, nthreads_y);
	cudaEvent_t start,end;
	picviz_verify_cuda(cudaEventCreate(&start));
	picviz_verify_cuda(cudaEventCreate(&end));


	// Compute number of blocks
	int nblocks = PVCuda::get_number_blocks();
	int nblocks_x = (width+nthreads_x-1)/nthreads_x;
	int nblocks_y = 1;
	PVLOG_INFO("Number of blocks: %d x %d\n", nblocks_x, nblocks_y);

	//int shared_size = nthreads_x*IMAGE_HEIGHT*sizeof(img_zbuffer_t);

	//picviz_verify_cuda(cudaFuncSetCacheConfig(bcicode_raster_unroll2<Bbits>, cudaFuncCachePreferL1));
	picviz_verify_cuda(cudaEventRecord(start, stream));
	bcicode_raster_unroll2<Bbits, reverse><<<dim3(nblocks_x,nblocks_y),dim3(nthreads_x, nthreads_y), 0, stream>>>((uint2*) device_codes, n, width, device_img, img_width, x_start, zoom_y);
	picviz_verify_cuda_kernel();
	picviz_verify_cuda(cudaEventRecord(end, stream));
	picviz_verify_cuda(cudaEventSynchronize(end));
	float time = 0;
	picviz_verify_cuda(cudaEventElapsedTime(&time, start, end));
	fprintf(stderr, "CUDA kernel time: %0.4f ms, BW: %0.4f MB/s\n", time, (double)(n*sizeof(PVBCICode<Bbits>))/(double)((time/1000.0)*1024.0*1024.0));
}

void show_codes_cuda10(PVParallelView::PVBCICode<10>* device_codes, uint32_t n, uint32_t width, uint32_t* device_img, uint32_t img_width, uint32_t x_start, const float zoom_y, cudaStream_t stream)
{
	show_codes_cuda<10, false>(device_codes, n, width, device_img, img_width, x_start, zoom_y, stream);
}

void show_codes_cuda11(PVParallelView::PVBCICode<11>* device_codes, uint32_t n, uint32_t width, uint32_t* device_img, uint32_t img_width, uint32_t x_start, const float zoom_y, cudaStream_t stream)
{
	show_codes_cuda<11, false>(device_codes, n, width, device_img, img_width, x_start, zoom_y, stream);
}

void show_codes_cuda11_reverse(PVParallelView::PVBCICode<11>* device_codes, uint32_t n, uint32_t width, uint32_t* device_img, uint32_t img_width, uint32_t x_start, const float zoom_y, cudaStream_t stream)
{
	show_codes_cuda<11, true>(device_codes, n, width, device_img, img_width, x_start, zoom_y, stream);
}
