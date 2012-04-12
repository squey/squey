#include <pvkernel/core/general.h>
#include <pvkernel/cuda/common.h>
#include <pvparallelview/common.h>
#include <pvparallelview/PVBCICode.h>
#include "bci_cuda.h"

#define NTHREADS_BLOCK 1024

using PVParallelView::PVBCICode;

struct img_zbuffer_t
{
	union {
		uint64_t int_v;
		struct {
			uint32_t pixel;
			uint32_t idx;
		} s;
	};
};

__global__ void bcicode_raster(uint2* bci_codes, uint32_t n, int32_t width, uint32_t* img_dst/*[width][IMAGE_HEIGHT]*/, int /*shared_size*/)
{
	// shared_size = blockDim.x*IMAGE_HEIGHT*sizeof(img_zbuffer_t)
	__shared__ uint2 shared_img[48*1024/sizeof(uint2)];

	// The x coordinate of the band this thread is responsible of
	const int band_x = threadIdx.x + blockIdx.x*blockDim.x;
	// Do this division once and for all
	// or not...
	const float alpha = (float)(width-band_x)/(float)width;

	// First stage is to clear shared memory
	for (int y = 0; y < IMAGE_HEIGHT; y++) {
		// idx is set to 0xFFFFFFFF (maximum value)
		// pixel is set to 0x00000000
		shared_img[threadIdx.x + y*blockDim.x] = make_uint2(0, 0xFFFFFFFF);
	}

	__syncthreads();

	const unsigned int size_grid = blockDim.y*gridDim.y;
	unsigned int idx_codes = threadIdx.y + blockIdx.y*blockDim.y;

	const unsigned int n_end = (n/(size_grid<<2))*(size_grid<<2);
	for (; idx_codes < n_end; idx_codes += size_grid<<2) {
		uint2 code0 = bci_codes[idx_codes];
		uint2 code1 = bci_codes[idx_codes+size_grid];
		uint2 code2 = bci_codes[idx_codes+2*size_grid];
		uint2 code3 = bci_codes[idx_codes+3*size_grid];

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
		//			uint32_t color: 11;
		//			uint32_t __reserved: 1;
		//		} s;
		//	};
		// }

		// Get l, r and color
		float l0 = (float) (code0.y & 0x3ff);
		float l1 = (float) (code1.y & 0x3ff);
		float l2 = (float) (code2.y & 0x3ff);
		float l3 = (float) (code3.y & 0x3ff);
		float r0 = (float) ((code0.y & 0xffc00)>>10);
		float r1 = (float) ((code1.y & 0xffc00)>>10);
		float r2 = (float) ((code2.y & 0xffc00)>>10);
		float r3 = (float) ((code3.y & 0xffc00)>>10);
		// TODO: compute real color
		//unsigned int color = (code.y & 0x7ff00000)>>20;

		// Compute the y coordinate for band_x
		int pixel_y0 = (int) (r0 + ((l0-r0)*alpha) + 0.5f);
		int pixel_y1 = (int) (r1 + ((l1-r1)*alpha) + 0.5f);
		int pixel_y2 = (int) (r2 + ((l2-r2)*alpha) + 0.5f);
		int pixel_y3 = (int) (r3 + ((l3-r3)*alpha) + 0.5f);
		unsigned int idx_shared_img0 = threadIdx.x + pixel_y0*blockDim.x;
		unsigned int idx_shared_img1 = threadIdx.x + pixel_y1*blockDim.x;
		unsigned int idx_shared_img2 = threadIdx.x + pixel_y2*blockDim.x;
		unsigned int idx_shared_img3 = threadIdx.x + pixel_y3*blockDim.x;

		// Set shared_img
		uint2 cur_shared_p = shared_img[idx_shared_img0];
		if (cur_shared_p.y > code0.x) {
			shared_img[idx_shared_img0] = make_uint2(0xFFFFFF00, code0.x);
		}
		cur_shared_p = shared_img[idx_shared_img1];
		if (cur_shared_p.y > code1.x) {
			shared_img[idx_shared_img1] = make_uint2(0xFFFFFF00, code1.x);
		}
		cur_shared_p = shared_img[idx_shared_img2];
		if (cur_shared_p.y > code2.x) {
			shared_img[idx_shared_img2] = make_uint2(0xFFFFFF00, code2.x);
		}
		cur_shared_p = shared_img[idx_shared_img3];
		if (cur_shared_p.y > code3.x) {
			shared_img[idx_shared_img3] = make_uint2(0xFFFFFF00, code3.x);
		}
	}
	for (; idx_codes < n; idx_codes += size_grid) {
		uint2 code0 = bci_codes[idx_codes];
		float l0 = (float) (code0.y & 0x3ff);
		float r0 = (float) ((code0.y & 0xffc00)>>10);
		int pixel_y0 = (int) (r0 + ((l0-r0)*alpha) + 0.5f);
		unsigned int idx_shared_img0 = threadIdx.x + pixel_y0*blockDim.x;
		uint2 cur_shared_p = shared_img[idx_shared_img0];
		if (cur_shared_p.y > code0.x) {
			shared_img[idx_shared_img0] = make_uint2(0xFFFFFF00, code0.x);
		}
	}

	__syncthreads();

	// Final stage is to commit the shared image into the global image
	for (int y = 0; y < IMAGE_HEIGHT; y++) {
		unsigned int pixel = shared_img[threadIdx.x + y*blockDim.x].x;
		img_dst[band_x + y*IMAGE_HEIGHT] = pixel;
	}
}

void show_codes_cuda(PVParallelView::PVBCICode* codes, uint32_t n, uint32_t width, uint32_t* img_dst/*[width][IMAGE_HEIGHT]*/)
{
	PVBCICode* device_codes;
	uint32_t* device_img;
	picviz_verify(sizeof(PVBCICode) == sizeof(uint64_t));

	picviz_verify_cuda(cudaMalloc(&device_codes, n*sizeof(PVBCICode)));
	picviz_verify_cuda(cudaMemcpy(device_codes, codes, n*sizeof(PVBCICode), cudaMemcpyHostToDevice));

	size_t simg = width*IMAGE_HEIGHT*sizeof(uint32_t);
	picviz_verify_cuda(cudaMalloc(&device_img, simg));
	picviz_verify_cuda(cudaMemset(device_img, 0, simg));
	
	cudaEvent_t start,end;
	picviz_verify_cuda(cudaEventCreate(&start));
	picviz_verify_cuda(cudaEventCreate(&end));

	// Compute number of threads per block
	int nthreads_x = picviz_min(width, PVCuda::get_shared_mem_size()/(IMAGE_HEIGHT*sizeof(img_zbuffer_t)));
	int nthreads_y = NTHREADS_BLOCK/nthreads_x;
	picviz_verify(nthreads_x*nthreads_y <= NTHREADS_BLOCK);
	PVLOG_INFO("Number threads per block: %d x %d\n", nthreads_x, nthreads_y);

	// Compute number of blocks
	int nblocks = PVCuda::get_number_blocks();
	int nblocks_x = width/nthreads_x;
	//int nblocks_y = ((nblocks_x+nblocks-1)/nblocks)*nblocks;
	int nblocks_y = 1;
	picviz_verify(nblocks_y > 0);
	PVLOG_INFO("Number of blocks: %d x %d\n", nblocks_x, nblocks_y);

	int shared_size = nthreads_x*IMAGE_HEIGHT*sizeof(img_zbuffer_t);

	picviz_verify_cuda(cudaEventRecord(start, 0));
	bcicode_raster<<<dim3(nblocks_x,nblocks_y),dim3(nthreads_x, nthreads_y)>>>((uint2*) device_codes, n, width, device_img, shared_size);
	picviz_verify_cuda_kernel();
	picviz_verify_cuda(cudaEventRecord(end, 0));
	picviz_verify_cuda(cudaEventSynchronize(end));

	picviz_verify_cuda(cudaMemcpy(img_dst, device_img, simg, cudaMemcpyDeviceToHost));

	picviz_verify_cuda(cudaFree(device_codes));
	picviz_verify_cuda(cudaFree(device_img));
	
	float time = 0;
	picviz_verify_cuda(cudaEventElapsedTime(&time, start, end));

	fprintf(stderr, "CUDA kernel time: %0.4f ms, BW: %0.4f MB/s\n", time, (double)(n*sizeof(PVBCICode))/(double)((time/1000.0)*1024.0*1024.0));
}
