/**
 * \file bci_cuda.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_BCICUDA_H
#define PVPARALLELVIEW_BCICUDA_H

// for GLuint
#include <GL/gl.h>

#include <pvparallelview/PVBCICode.h>

void show_codes_cuda10(PVParallelView::PVBCICode<10>* device_codes, uint32_t n, uint32_t width, uint32_t* device_img, uint32_t img_width, uint32_t x_start, const float zoom_y = 1.0f, cudaStream_t stream = NULL);
void show_codes_cuda11(PVParallelView::PVBCICode<11>* device_codes, uint32_t n, uint32_t width, uint32_t* device_img, uint32_t img_width, uint32_t x_start, const float zoom_y = 1.0f, cudaStream_t stream = NULL);
void show_codes_cuda11_reverse(PVParallelView::PVBCICode<11>* device_codes, uint32_t n, uint32_t width, uint32_t* device_img, uint32_t img_width, uint32_t x_start, const float zoom_y = 1.0f, cudaStream_t stream = NULL);

#endif
