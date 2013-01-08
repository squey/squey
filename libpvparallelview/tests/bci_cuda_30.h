/**
 * \file bci_cuda.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_BCICUDA_TEST_30_H
#define PVPARALLELVIEW_BCICUDA_TEST_30_H

#include <pvkernel/cuda/common.h>
#include <pvparallelview/PVBCICode.h>

// For benchmarking pruposes
float show_and_perf_codes_cuda10_test(PVParallelView::PVBCICode<10>* device_codes, uint32_t n, uint32_t width, uint32_t* device_img, uint32_t img_width, uint32_t x_start, const float zoom_y = 1.0f, cudaStream_t stream = NULL, double* bw = NULL);
float show_and_perf_codes_cuda11_test(PVParallelView::PVBCICode<11>* device_codes, uint32_t n, uint32_t width, uint32_t* device_img, uint32_t img_width, uint32_t x_start, const float zoom_y = 1.0f, cudaStream_t stream = NULL, double* bw = NULL);

#endif
