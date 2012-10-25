/**
 * \file multiple_zones_cuda.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/cuda/common.h>
#include <pvparallelview/common.h>
#include <pvkernel/core/PVHSVColor.h>
#include <pvparallelview/PVBCICode.h>
#include "multiple_zones_cuda.h"
#include <pvparallelview/simple_lines_int_view.h>

#include <iostream>

#define WIDTH 1024
#define ZONE_WIDTH 1024

#include <QApplication>
#include <QMainWindow>

void stream_callback(cudaStream_t stream, cudaError_t status, void* userData)
{
	printf("stream %d is finished\n", stream);
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " nlines" << " [width]" <<  " [zone_width]" << std::endl;
		return 1;
	}

	size_t width = WIDTH;
	size_t zone_width = ZONE_WIDTH;
	if (argc >= 3) {
		width = atoll(argv[2]);
	}
	if (argc >= 4) {
		zone_width = atoll(argv[3]);
	}

	srand(0);

	QApplication app(argc, argv);
	PVCuda::init_cuda();

	// Enable peer-to-peer access
	picviz_verify_cuda(cudaSetDevice(1));
	picviz_verify_cuda(cudaDeviceEnablePeerAccess(0, 0));

	size_t n = atoll(argv[1]);

	PVParallelView::PVBCICode<NBITS_INDEX>* host_codes = PVParallelView::PVBCICode<NBITS_INDEX>::allocate_codes(n);
	PVParallelView::PVBCICode<NBITS_INDEX>::init_random_codes(host_codes, n);

	uint32_t* device_img = init_cuda_image(width);
	PVParallelView::PVBCICode<NBITS_INDEX>* device_codes[2];
	device_codes[0] = init_cuda_codes(0);
	device_codes[1] = init_cuda_codes(1);

	const int nzones = width/zone_width; 
	cudaStream_t *cstreams = new cudaStream_t[nzones];
	for (int z = 0; z < nzones; z++) {
		picviz_verify_cuda(cudaSetDevice(z & 1));
		picviz_verify_cuda(cudaStreamCreate(&cstreams[z]));
	}
	cudaEvent_t start,end;
	cudaEventCreate(&start); cudaEventCreate(&end);
	picviz_verify_cuda(cudaEventRecord(start, 0));
	for (int z = 0; z < nzones; z++) {
		int device = z & 1;
		cudaSetDevice(device);
		printf("Launching kernel on stream %d w/ device %d\n", cstreams[z], device);
		copy_codes_to_cuda(device_codes[device], host_codes, n, cstreams[z]);
		show_codes_cuda(device_codes[device], n, zone_width, device_img, width, z*zone_width, cstreams[z]);
		picviz_verify_cuda(cudaStreamAddCallback(cstreams[z], &stream_callback, NULL, 0));

	}
	picviz_verify_cuda(cudaEventRecord(end, 0));
	picviz_verify_cuda(cudaEventSynchronize(end));

	float time;
	cudaEventElapsedTime(&time, start, end);

	cudaEventDestroy(start); cudaEventDestroy(end);
	for (int z = 0; z < nzones; z++) {
		cudaStreamDestroy(cstreams[z]);
	}

	fprintf(stderr, "Time for drawing %d zones of %lu pixels: %0.4f ms\n", nzones, zone_width, time);

	uint32_t* img = new uint32_t[width*IMAGE_HEIGHT];
	copy_img_cuda(img, device_img, width);
	write(4, img, width*IMAGE_HEIGHT*sizeof(uint32_t));
	delete [] img;

	//show_codes(codes, n);
	//app.exec();

	PVParallelView::PVBCICode<NBITS_INDEX>::free_codes(host_codes);
	free_cuda_buffer(device_img);
	free_cuda_buffer(device_codes[0], 0);
	free_cuda_buffer(device_codes[1], 1);

	cudaDeviceReset();

	return 0;
}
