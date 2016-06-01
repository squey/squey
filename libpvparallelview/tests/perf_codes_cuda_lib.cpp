/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/inendi_bench.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVBCIBackendImage.h>
#include <pvparallelview/PVBCIDrawingBackendCUDA.h>
#include <pvparallelview/cuda/bci_cuda.h>
#include <pvparallelview/PVTools.h>

#include "helpers.h"

constexpr size_t default_width = 1024;
constexpr size_t default_bbits = 10;

template <size_t Bbits>
PVParallelView::PVBCIBackendImage_p do_test(size_t n, size_t width, int pattern)
{
	PVParallelView::PVBCIDrawingBackendCUDA& backend_cuda =
	    PVParallelView::PVBCIDrawingBackendCUDA::get();

	// Init "n" BCI codes.
	PVParallelView::PVBCICode<Bbits>* codes = PVParallelView::PVBCICode<Bbits>::allocate_codes(n);
	PVParallelView::PVBCIPatterns<Bbits>::init_codes_pattern(codes, n, pattern);

	// Allocate memory for a cuda image.
	PVParallelView::PVBCIBackendImage_p dst_img = backend_cuda.create_image(width, Bbits);
	PVParallelView::PVBCIBackendImageCUDA* dst_img_cuda =
	    dynamic_cast<PVParallelView::PVBCIBackendImageCUDA*>(dst_img.get());
	assert(dst_img_cuda);

	// Transfert BCI code from host to device
	PVParallelView::PVBCICodeBase* dev_codes;
	inendi_verify_cuda(cudaMalloc(&dev_codes, n * sizeof(PVParallelView::PVBCICodeBase)));
	inendi_verify_cuda(cudaMemcpy(dev_codes, codes, n * sizeof(codes), cudaMemcpyHostToDevice));
	double bw;
	float time;
	switch (Bbits) {
	case 10:
		time = show_and_perf_codes_cuda10(&dev_codes->as_10, n, width, dst_img_cuda->device_img(),
		                                  width, 0, 1.0f, NULL, &bw);
		break;
	case 11:
		time = show_and_perf_codes_cuda11(&dev_codes->as_11, n, width, dst_img_cuda->device_img(),
		                                  width, 0, 1.0f, NULL, &bw);
		break;
	}

	std::cout << "Kernel launch in " << time << "ms, BW = " << bw << "MB/s" << std::endl;

	// Get image from device to host
	dst_img_cuda->copy_device_to_host();

	// Transform Image in QImage.
	QImage img(dst_img->qimage());

	// TODO : remove?
	write(4, img.constBits(), img.height() * img.width() * sizeof(uint32_t));

	return dst_img;
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " nlines"
		          << " [width] [pattern] [bbits]" << std::endl;
		const char* const* patterns =
		    PVParallelView::PVBCIPatterns<default_bbits>::get_patterns_string();
		std::cerr << "where pattern is one of the following:" << std::endl;
		for (int i = 0; i < PVParallelView::PVBCIPatterns<default_bbits>::get_number_patterns();
		     i++) {
			std::cerr << i << "\t-\t" << patterns[i] << std::endl;
		}
		std::cerr << "and bbits is 10 (1024 image height) or 11 (2048 image height)." << std::endl;
		return 1;
	}

	size_t width = default_width;
	int pattern = 0;
	int bbits = default_bbits;
	if (argc >= 3) {
		width = atoll(argv[2]);
	}
	if (argc >= 4) {
		pattern = atoi(argv[3]);
	}
	if (argc >= 5) {
		bbits = atoi(argv[4]);
		if (bbits != 10 && bbits != 11) {
			std::cerr << "bbits must be 10 or 11 !" << std::endl;
			return 1;
		}
	}

	size_t n = atoll(argv[1]);

	PVParallelView::PVBCIBackendImage_p dst_img;
	switch (bbits) {
	case 10:
		dst_img = do_test<10>(n, width, pattern);
		break;
	case 11:
		dst_img = do_test<11>(n, width, pattern);
		break;
	}

	return 0;
}
