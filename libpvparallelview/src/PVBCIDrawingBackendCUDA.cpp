/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/inendi_bench.h>
#include <pvparallelview/common.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVBCIDrawingBackendCUDA.h>
#include <pvparallelview/cuda/bci_cuda.h>

/******************************************************************************
 *
 * cuda_kernel
 *
 *****************************************************************************/

template <size_t Bbits>
struct cuda_kernel;

template <>
struct cuda_kernel<10> {
	static inline void launch(PVParallelView::PVBCICode<10>* device_codes,
	                          uint32_t n,
	                          uint32_t width,
	                          uint32_t* device_img,
	                          uint32_t img_width,
	                          uint32_t x_start,
	                          const float zoom_y,
	                          cudaStream_t stream = NULL,
	                          bool reverse = false)
	{
		assert(reverse == false);
		(void)reverse; // remove warning "unused parameter"
		show_codes_cuda10(device_codes, n, width, device_img, img_width, x_start, zoom_y, stream);
	}
};

template <>
struct cuda_kernel<11> {
	static inline void launch(PVParallelView::PVBCICode<11>* device_codes,
	                          uint32_t n,
	                          uint32_t width,
	                          uint32_t* device_img,
	                          uint32_t img_width,
	                          uint32_t x_start,
	                          const float zoom_y,
	                          cudaStream_t stream = NULL,
	                          bool reverse = false)
	{
		if (reverse) {
			show_codes_cuda11_reverse(device_codes, n, width, device_img, img_width, x_start,
			                          zoom_y, stream);
		} else {
			show_codes_cuda11(device_codes, n, width, device_img, img_width, x_start, zoom_y,
			                  stream);
		}
	}
};

/******************************************************************************
 *
 * BackendImageCUDA
 *
 *****************************************************************************/

PVParallelView::PVBCIBackendImageCUDA::PVBCIBackendImageCUDA(const uint32_t width,
                                                             uint8_t height_bits,
                                                             const int cuda_device,
                                                             cudaStream_t stream)
    : PVBCIBackendImage(width, height_bits), _cuda_device(cuda_device)
{
	size_t simg = PVBCIBackendImage::size_pixel();
	//_host_img = pixel_allocator().allocate(simg);
	inendi_verify_cuda(cudaHostAlloc(&_host_img, simg * sizeof(pixel_t), cudaHostAllocPortable));
	set_current_device();
	inendi_verify_cuda(cudaMalloc(&_device_img, simg * sizeof(pixel_t)));
	inendi_verify_cuda(cudaMemsetAsync(_device_img, 0, simg * sizeof(pixel_t), stream));
	_org_width = width;
}

PVParallelView::PVBCIBackendImageCUDA::~PVBCIBackendImageCUDA()
{
	inendi_verify_cuda(cudaFreeHost(_host_img));
	inendi_verify_cuda(cudaFree(_device_img));
}

/******************************************************************************
 *
 * DrawingBackendCUDA
 *
 *****************************************************************************/

PVParallelView::PVBCIDrawingBackendCUDA::PVBCIDrawingBackendCUDA()
{
	// List all usable cuda engines and create stream and appropriate structure
	std::vector<int> list_ids;
	PVCuda::visit_usable_cuda_devices([&](int id) {
		inendi_verify_cuda(cudaSetDevice(id));
		inendi_verify_cuda(cudaDeviceReset());
		inendi_verify_cuda(cudaGLSetGLDevice(id));
		inendi_verify_cuda(cudaDeviceReset());
		// Set scheduling to yield, as we need all processes!
		cudaSetDeviceFlags(cudaDeviceScheduleYield | cudaDeviceMapHost);
		list_ids.push_back(id);

		device_t dev;
		inendi_verify_cuda(
		    cudaMalloc(&dev.device_codes, PVParallelView::MaxBciCodes * sizeof(PVBCICodeBase)));
		inendi_verify_cuda(cudaStreamCreate(&dev.stream));
		this->_devices.insert(std::make_pair(id, dev));
	});

	// Enable full P2P access!
	if (list_ids.size() > 1) {
		for (size_t i = 0; i < list_ids.size(); i++) {
			for (size_t j = i + 1; j < list_ids.size(); j++) {
				cudaSetDevice(i);
				cudaDeviceEnablePeerAccess(j, 0);
				cudaSetDevice(j);
				cudaDeviceEnablePeerAccess(i, 0);
			}
		}
	}

	_last_image_dev = _devices.begin();
}

PVParallelView::PVBCIDrawingBackendCUDA::~PVBCIDrawingBackendCUDA()
{
	for (auto& device : _devices) {
		cudaSetDevice(device.first);
		inendi_verify_cuda(cudaFree(device.second.device_codes));
		inendi_verify_cuda(cudaStreamDestroy(device.second.stream));
	}
}

PVParallelView::PVBCIDrawingBackendCUDA& PVParallelView::PVBCIDrawingBackendCUDA::get()
{
	static PVBCIDrawingBackendCUDA backend;
	return backend;
}

PVParallelView::PVBCIBackendImage_p
PVParallelView::PVBCIDrawingBackendCUDA::create_image(size_t img_width, uint8_t height_bits)
{
	assert(_devices.size() >= 1);
	if (_last_image_dev == _devices.end()) {
		_last_image_dev = _devices.begin();
	}

	// Create image on a device in a round robin way
	int dev = _last_image_dev->first;
	PVBCIBackendImage_p ret(
	    new PVBCIBackendImageCUDA(img_width, height_bits, dev, _last_image_dev->second.stream));
	++_last_image_dev;
	return ret;
}

void PVParallelView::PVBCIDrawingBackendCUDA::operator()(PVBCIBackendImage_p& dst_img,
                                                         size_t x_start,
                                                         size_t width,
                                                         PVBCICodeBase* codes,
                                                         size_t n,
                                                         const float zoom_y,
                                                         bool reverse,
                                                         std::function<void()> const& render_done)
{
#ifdef NDEBUG
	backend_image_t* dst_img_cuda = static_cast<backend_image_t*>(dst_img.get());
#else
	backend_image_t* dst_img_cuda = dynamic_cast<backend_image_t*>(dst_img.get());
	assert(dst_img_cuda != NULL);
#endif
	// assert(x_start + width <= (size_t) dst_img_cuda->width());

	// Set current device according to dst_img
	int dev_id = dst_img_cuda->get_cuda_device();
	dst_img_cuda->set_current_device();

	device_t& dev = _devices[dev_id];
	cuda_job_data* data = new cuda_job_data();
	data->done_function = render_done;
	cudaStream_t stream = dev.stream;

	inendi_verify_cuda(cudaMemcpyAsync(dev.device_codes, codes, n * sizeof(codes),
	                                   cudaMemcpyHostToDevice, stream));
	switch (dst_img->height_bits()) {
	case 10:
		cuda_kernel<10>::launch(&dev.device_codes->as_10, n, width, dst_img_cuda->device_img(),
		                        dst_img_cuda->org_width(), x_start, zoom_y, stream, reverse);
		break;
	case 11:
		cuda_kernel<11>::launch(&dev.device_codes->as_11, n, width, dst_img_cuda->device_img(),
		                        dst_img_cuda->org_width(), x_start, zoom_y, stream, reverse);
		break;
	default:
		assert(false);
		break;
	};
	inendi_verify_cuda_kernel();
	dst_img_cuda->copy_device_to_host(stream);
	inendi_verify_cuda(cudaStreamAddCallback(
	    stream, &PVBCIDrawingBackendCUDA::image_rendered_and_copied_callback, (void*)data, 0));
}

void PVParallelView::PVBCIDrawingBackendCUDA::image_rendered_and_copied_callback(
    cudaStream_t /*stream*/, cudaError_t /*status*/, void* data_)
{
	assert(data_);

	cuda_job_data* data = reinterpret_cast<cuda_job_data*>(data_);

	// Call termination function
	if (data->done_function) {
		(data->done_function)();
	}

	delete data;
}

void PVParallelView::PVBCIDrawingBackendCUDA::wait_all() const
{
	// Wait all GPUs!
	for (auto& device : _devices) {
		cudaSetDevice(device.first);
		cudaDeviceSynchronize();
	}
}

PVParallelView::PVBCICodeBase* PVParallelView::PVBCIDrawingBackendCUDA::allocate_bci(size_t n)
{
	PVParallelView::PVBCICodeBase* ret;
	inendi_verify_cuda(
	    cudaHostAlloc(&ret, n * sizeof(PVParallelView::PVBCICodeBase), cudaHostAllocPortable));
	return ret;
}

void PVParallelView::PVBCIDrawingBackendCUDA::free_bci(PVParallelView::PVBCICodeBase* buf)
{
	inendi_verify_cuda(cudaFreeHost(buf));
}
