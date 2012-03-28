#ifndef OCL_RASTER_H
#define OCL_RASTER_H

#define LOCAL_IDX_WIDTH 32
#define LOCAL_IDX_HEIGHT 32

#define IMAGE_WIDTH 2048
#define IMAGE_HEIGHT 2048

#define SIZE_GLOBAL_IDX_TABLE (IMAGE_WIDTH*IMAGE_HEIGHT*sizeof(unsigned int))
#define SIZE_LOCAL_IDX_TABLE (LOCAL_IDX_WIDTH*LOCAL_IDX_HEIGHT*sizeof(unsigned int))

#define L3_CACHE_SIZE (12*1024*1024)
#define NLINES_PER_KERNEL (L3_CACHE_SIZE/(2*sizeof(float)))
//#define NLINES_PER_KERNEL (100000000)

void ocl_raster(const char* kernels_file, const float* yl, const float* yr, const size_t n, unsigned int* img_idxes, const float zoom_x, const float zoom_y);

#endif
