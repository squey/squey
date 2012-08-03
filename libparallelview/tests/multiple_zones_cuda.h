/**
 * \file multiple_zones_cuda.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef MULTIPLE_ZONES_CUDA_H
#define MULTIPLE_ZONES_CUDA_H

#include <pvkernel/cuda/common.h>

namespace PVParallelView {
class PVBCICode;
}

uint32_t* init_cuda_image(uint32_t img_width, int device = 0);
PVParallelView::PVBCICode<NBITS_INDEX>* init_cuda_codes(int device = 0);
void free_cuda_buffer(void* buf, int device = 0);
void show_codes_cuda(PVParallelView::PVBCICode<NBITS_INDEX>* device_codes, uint32_t n, uint32_t width, uint32_t* device_img, uint32_t img_width, uint32_t x_start, cudaStream_t stream = NULL);
void copy_img_cuda(uint32_t* host_img, uint32_t* device_img, uint32_t width);
void copy_codes_to_cuda(PVParallelView::PVBCICode<NBITS_INDEX>* host_codes, PVParallelView::PVBCICode<NBITS_INDEX>* device_codes, size_t n, cudaStream_t stream = NULL);

#endif
