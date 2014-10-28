#include <pvkernel/core/picviz_bench.h>
#include <pvparallelview/common.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVBCIDrawingBackendCUDA.h>
#include <pvparallelview/cuda/bci_cuda.h>

#include <npp.h> // NVIDIA's NPP library, for image resizing

#include <iostream>

PVParallelView::PVBCIDrawingBackendCUDA* PVParallelView::PVBCIDrawingBackendCUDA::_instance = nullptr;

template <size_t Bbits>
struct cuda_kernel;

template <>
struct cuda_kernel<10>
{
	static inline void launch(PVParallelView::PVBCICode<10>* device_codes, uint32_t n, uint32_t width, uint32_t* device_img, uint32_t img_width, uint32_t x_start, const float zoom_y, cudaStream_t stream = NULL, bool reverse = false)
	{
		assert(reverse == false);
		(void)reverse; // remove warning "unused parameter"
		show_codes_cuda10(device_codes, n, width, device_img, img_width, x_start, zoom_y, stream);
	}
};

template <>
struct cuda_kernel<11>
{
	static inline void launch(PVParallelView::PVBCICode<11>* device_codes, uint32_t n, uint32_t width, uint32_t* device_img, uint32_t img_width, uint32_t x_start, const float zoom_y, cudaStream_t stream = NULL, bool reverse = false)
	{
		if (reverse) {
			show_codes_cuda11_reverse(device_codes, n, width, device_img, img_width, x_start, zoom_y, stream);
		}
		else {
			show_codes_cuda11(device_codes, n, width, device_img, img_width, x_start, zoom_y, stream);
		}
	}
};

PVParallelView::PVBCIBackendImageCUDA::PVBCIBackendImageCUDA(const uint32_t width, uint8_t height_bits, const int cuda_device, cudaStream_t stream):
	PVBCIBackendImage(width, height_bits),
	_cuda_device(cuda_device)
{
	size_t simg = PVBCIBackendImage::size_pixel();
	//_host_img = pixel_allocator().allocate(simg);
	picviz_verify_cuda(cudaHostAlloc(&_host_img, simg*sizeof(pixel_t), cudaHostAllocPortable));
	set_current_device();
	picviz_verify_cuda(cudaMalloc(&_device_img, simg*sizeof(pixel_t)));
	picviz_verify_cuda(cudaMemsetAsync(_device_img, 0, simg*sizeof(pixel_t), stream));
	_org_width = width;
}

PVParallelView::PVBCIBackendImageCUDA::~PVBCIBackendImageCUDA()
{
	//size_t simg = PVBCIBackendImage::size_pixel();
	//pixel_allocator().deallocate(_host_img, simg);
	picviz_verify_cuda(cudaFreeHost(_host_img));
	picviz_verify_cuda(cudaFree(_device_img));
}

void PVParallelView::PVBCIBackendImageCUDA::resize_width(PVBCIBackendImage& dst, const uint32_t width) const
{
	assert(org_width() % 4 == 0);

	PVBCIBackendImageCUDA* dst_img = static_cast<PVBCIBackendImageCUDA*>(&dst);
	assert(dst_img->org_width() % 4 == 0);

	set_current_device();

	cudaEvent_t end;
	cudaEventCreate(&end);

	// NPP resize method
	NppiRect rorg;
	rorg.x = 0; rorg.y = 0;
	rorg.width = PVBCIBackendImage::width(); rorg.height = PVBCIBackendImage::height();
	NppiSize sorg, sdst;
	sorg.width = PVBCIBackendImage::width(); sorg.height = PVBCIBackendImage::height();
	sdst.width = width; sdst.height = sorg.height;
	nppiResize_8u_C4R((const Npp8u*) device_img(), sorg, org_width()*sizeof(pixel_t), rorg, (Npp8u*) dst_img->device_img(), dst_img->org_width()*sizeof(pixel_t), sdst, (double)width/(double)sorg.width, 1.0, NPPI_INTER_NN);

	// wait for the end
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventDestroy(end);

	dst_img->set_width(width);
	dst_img->set_current_device();
	dst_img->copy_device_to_host();
}

PVParallelView::PVBCIDrawingBackendCUDA::PVBCIDrawingBackendCUDA()
{
	PVCuda::init_cuda();

	// List all usable cuda engines and create stream and appropriate structure
	std::vector<int> list_ids;
	PVCuda::visit_usable_cuda_devices([&](int id)
			{
				picviz_verify_cuda(cudaSetDevice(id));
				picviz_verify_cuda(cudaDeviceReset());
				picviz_verify_cuda(cudaGLSetGLDevice(id));
				picviz_verify_cuda(cudaDeviceReset());
				// Set scheduling to yield, as we need all processes!
				cudaSetDeviceFlags(cudaDeviceScheduleYield | cudaDeviceMapHost);
				list_ids.push_back(id);

				device_t dev;
				picviz_verify_cuda(cudaMalloc(&dev.device_codes, PVParallelView::MaxBciCodes * sizeof(PVBCICodeBase)));
				picviz_verify_cuda(cudaStreamCreate(&dev.stream));
				this->_devices.insert(std::make_pair(id, dev));
			});

	// Enable full P2P access!
	if (list_ids.size() > 1) {
		std::sort(list_ids.begin(), list_ids.end());
		do {
			cudaSetDevice(*list_ids.begin());
			cudaDeviceEnablePeerAccess(*(list_ids.begin()+1), 0);
		}
		while (std::next_permutation(list_ids.begin(), list_ids.end()));
	}


	/*
	// Init stream pools
	const size_t ndevs = _devices.size();
	const size_t streams_per_dev = (BCI_BUFFERS_COUNT+ndevs-1)/ndevs;

	decltype(_devices)::iterator it;
	for (it = _devices.begin(); it != _devices.end(); it++) {
		it->second.streams.init(streams_per_dev);
	}*/

	_last_image_dev = _devices.begin();
}

PVParallelView::PVBCIDrawingBackendCUDA::~PVBCIDrawingBackendCUDA()
{
	decltype(_devices)::const_iterator it;
	for (it = _devices.begin(); it != _devices.end(); it++) {
		cudaSetDevice(it->first);
		picviz_verify_cuda(cudaFree(it->second.device_codes));
		picviz_verify_cuda(cudaStreamDestroy(it->second.stream));
	}
}

PVParallelView::PVBCIDrawingBackendCUDA& PVParallelView::PVBCIDrawingBackendCUDA::get()
{
	if (_instance == nullptr) {
		_instance = new PVBCIDrawingBackendCUDA();
	}
	return *_instance;
}

void PVParallelView::PVBCIDrawingBackendCUDA::release()
{
	if (_instance) {
		delete _instance;
	}
}

PVParallelView::PVBCIBackendImage_p PVParallelView::PVBCIDrawingBackendCUDA::create_image(size_t img_width, uint8_t height_bits) const
{
	assert(_devices.size() >= 1);
	if (_last_image_dev == _devices.end()) {
		_last_image_dev = _devices.begin();
	}

	int dev = _last_image_dev->first;
	PVBCIBackendImage_p ret(new PVBCIBackendImageCUDA(img_width, height_bits, dev, _last_image_dev->second.stream));
	++_last_image_dev;
	return ret;
}

PVParallelView::PVBCIBackendImage_p PVParallelView::PVBCIDrawingBackendCUDA::create_image_on_same_device(size_t img_width, uint8_t height_bits, backend_image_t const& ref) const
{
	assert(_devices.size() >= 1);

	PVBCIBackendImageCUDA const* ref_cuda = dynamic_cast<PVBCIBackendImageCUDA const*>(&ref);
	int dev;
	if (ref_cuda) {
		dev = ref_cuda->get_cuda_device();
	}
	else {
		if (_last_image_dev == _devices.end()) {
			_last_image_dev = _devices.begin();
		}
		dev = _last_image_dev->first;
		++_last_image_dev;
	}
	PVBCIBackendImage_p ret(new PVBCIBackendImageCUDA(img_width, height_bits, dev, _devices[dev].stream));
	return ret;
}

void PVParallelView::PVBCIDrawingBackendCUDA::operator()(PVBCIBackendImage_p& dst_img, size_t x_start, size_t width, PVBCICodeBase* codes, size_t n, const float zoom_y, bool reverse, std::function<void()> const& render_done)
{
#ifdef NBDEUG
	backend_image_t* dst_img_cuda = static_cast<backend_image_t*>(dst_img.get());
#else
	backend_image_t* dst_img_cuda = dynamic_cast<backend_image_t*>(dst_img.get());
	assert(dst_img_cuda != NULL);
#endif
	//assert(x_start + width <= (size_t) dst_img_cuda->width());

	// Set current device according to dst_img
	int dev_id = dst_img_cuda->get_cuda_device();
	dst_img_cuda->set_current_device();

	device_t& dev = _devices[dev_id];
	cuda_job_data* data = new cuda_job_data();
	data->done_function = render_done;
	cudaStream_t stream = dev.stream;

	picviz_verify_cuda(cudaMemcpyAsync(dev.device_codes, codes, n*sizeof(codes), cudaMemcpyHostToDevice, stream));
	switch (dst_img->height_bits()) {
		case 10:
			cuda_kernel<10>::launch(&dev.device_codes->as<10>(), n, width, dst_img_cuda->device_img(), dst_img_cuda->org_width(), x_start, zoom_y, stream, reverse);
			break;
		case 11:
			cuda_kernel<11>::launch(&dev.device_codes->as<11>(), n, width, dst_img_cuda->device_img(), dst_img_cuda->org_width(), x_start, zoom_y, stream, reverse);
			break;
		default:
			assert(false);
			break;
	};
	picviz_verify_cuda_kernel();
	dst_img_cuda->copy_device_to_host(stream);
	picviz_verify_cuda(cudaStreamAddCallback(stream, &PVBCIDrawingBackendCUDA::image_rendered_and_copied_callback, (void*) data, 0));
}

void PVParallelView::PVBCIDrawingBackendCUDA::image_rendered_and_copied_callback(cudaStream_t /*stream*/, cudaError_t /*status*/, void* data_)
{
	assert(data_);

	cuda_job_data* data = reinterpret_cast<cuda_job_data*>(data_);
	
	// Get back stream
	//data->stream_pool->return_stream(stream);

	// Call termination function
	if (data->done_function) {
		(data->done_function)();
	}

	delete data;
}

void PVParallelView::PVBCIDrawingBackendCUDA::wait_all()
{
	// Wait all GPUs!
	decltype(_devices)::const_iterator it;
	for (it = _devices.begin(); it != _devices.end(); it++) {
		cudaSetDevice(it->first);
		cudaDeviceSynchronize();
	}
}

PVParallelView::PVBCICodeBase* PVParallelView::PVBCIDrawingBackendCUDA::allocate_bci(size_t n)
{
	PVParallelView::PVBCICodeBase* ret;
	picviz_verify_cuda(cudaHostAlloc(&ret, n*sizeof(PVParallelView::PVBCICodeBase), cudaHostAllocPortable));
	return ret;
}

void PVParallelView::PVBCIDrawingBackendCUDA::free_bci(PVParallelView::PVBCICodeBase* buf)
{
	picviz_verify_cuda(cudaFreeHost(buf));
}
