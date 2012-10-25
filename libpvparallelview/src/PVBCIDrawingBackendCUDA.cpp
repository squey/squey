#include <pvparallelview/common.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVBCIDrawingBackendCUDA.h>
#include <pvparallelview/cuda/bci_cuda.h>

#include <npp.h> // NVIDIA's NPP library, for image resizing


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

PVParallelView::PVBCIBackendImageCUDA::PVBCIBackendImageCUDA(const uint32_t width, uint8_t height_bits, const int cuda_device):
	PVBCIBackendImage(width, height_bits),
	_cuda_device(cuda_device)
{
	size_t simg = PVBCIBackendImage::size_pixel();
	_host_img = pixel_allocator().allocate(simg);
	set_current_device();
	picviz_verify_cuda(cudaMalloc(&_device_img, simg*sizeof(pixel_t)));
	picviz_verify_cuda(cudaMemset(_device_img, 0, simg*sizeof(pixel_t)));
	_org_width = width;
}

PVParallelView::PVBCIBackendImageCUDA::~PVBCIBackendImageCUDA()
{
	size_t simg = PVBCIBackendImage::size_pixel();
	pixel_allocator().deallocate(_host_img, simg);
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
	dst_img->copy_device_to_host();
}

PVParallelView::PVBCIDrawingBackendCUDA::PVBCIDrawingBackendCUDA()
{
	// List all usable cuda engines and create stream and appropriate structure
	PVCuda::visit_usable_cuda_devices([&](int id)
			{
				cudaSetDevice(id);

				device_t dev;
				picviz_verify_cuda(cudaMalloc(&dev.device_codes, PVParallelView::MaxBciCodes * sizeof(PVBCICodeBase)));
				picviz_verify_cuda(cudaStreamCreate(&dev.stream));
				this->_devices.insert(std::make_pair(id, dev));
			});
	_last_image_dev = _devices.begin();
}

PVParallelView::PVBCIDrawingBackendCUDA::~PVBCIDrawingBackendCUDA()
{
	decltype(_devices)::const_iterator it;
	for (it = _devices.begin(); it != _devices.end(); it++) {
		cudaSetDevice(it->first);
		picviz_verify_cuda(cudaStreamDestroy(it->second.stream));
		picviz_verify_cuda(cudaFree(it->second.device_codes));
	}
}

PVParallelView::PVBCIBackendImage_p PVParallelView::PVBCIDrawingBackendCUDA::create_image(size_t img_width, uint8_t height_bits) const
{
	assert(_devices.size() >= 1);
	if (_last_image_dev == _devices.end()) {
		_last_image_dev = _devices.begin();
	}

	//int dev = _last_image_dev->first;
	// AG: tofix: crashes with GTX 690!!
	PVBCIBackendImage_p ret(new PVBCIBackendImageCUDA(img_width, height_bits, 0));
	//PVBCIBackendImage_p<Bbits> ret(new PVBCIBackendImageCUDA<Bbits>(img_width, dev));
	++_last_image_dev;
	return ret;
}

void PVParallelView::PVBCIDrawingBackendCUDA::operator()(PVBCIBackendImage& dst_img, size_t x_start, size_t width, PVBCICodeBase* codes, size_t n, const float zoom_y, bool reverse, std::function<void()> const& /*render_done*/)
{
#ifdef NBDEUG
	backend_image_t* dst_img_cuda = static_cast<backend_image_t*>(&dst_img);
#else
	backend_image_t* dst_img_cuda = dynamic_cast<backend_image_t*>(&dst_img);
	assert(dst_img_cuda != NULL);
#endif
	assert(x_start + width <= (size_t) dst_img_cuda->width());

	// Set current device according to dst_img
	int dev_id = dst_img_cuda->get_cuda_device();
	dst_img_cuda->set_current_device();
	assert(_devices.find(dev_id) != _devices.end());

	device_t const& dev = _devices[dev_id];

	cudaEvent_t end;
	picviz_verify_cuda(cudaEventCreateWithFlags(&end, cudaEventBlockingSync | cudaEventDisableTiming));

	picviz_verify_cuda(cudaMemcpyAsync(dev.device_codes, codes, n*sizeof(codes), cudaMemcpyHostToDevice, dev.stream));
	switch (dst_img.height_bits()) {
		case 10:
			cuda_kernel<10>::launch(&dev.device_codes->as<10>(), n, width, dst_img_cuda->device_img(), dst_img_cuda->org_width(), x_start, zoom_y, dev.stream, reverse);
			break;
		case 11:
			cuda_kernel<11>::launch(&dev.device_codes->as<11>(), n, width, dst_img_cuda->device_img(), dst_img_cuda->org_width(), x_start, zoom_y, dev.stream, reverse);
			break;
		default:
			assert(false);
			break;
	};
	dst_img_cuda->copy_device_to_host(dev.stream);
	picviz_verify_cuda(cudaEventRecord(end, dev.stream));
	picviz_verify_cuda(cudaEventSynchronize(end));
	picviz_verify_cuda(cudaEventDestroy(end));
}
