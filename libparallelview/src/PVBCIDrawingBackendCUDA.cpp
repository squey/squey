#include <pvparallelview/PVBCIBackendImage.h>
#include <pvparallelview/PVBCIDrawingBackendCUDA.h>
#include <pvparallelview/cuda/bci_cuda.h>

#include <QImage>
#include <cassert>

namespace PVParallelView { namespace __impl {

class PVBCIBackendImageCUDA: public PVParallelView::PVBCIBackendImage
{
	friend class PVParallelView::PVBCIDrawingBackendCUDA;
	
	typedef PVParallelView::PVBCIDrawingBackendCUDA::pixel_t pixel_t;
	typedef PVParallelView::PVBCIDrawingBackendCUDA::pixel_pointer_t pixel_pointer_t;

	typedef std::allocator<pixel_t> pixel_allocator;

protected:
	PVBCIBackendImageCUDA(uint32_t width):
		PVBCIBackendImage(width)
	{
		size_t simg = size_pixel();
		_host_img = pixel_allocator().allocate(simg);
		picviz_verify_cuda(cudaMalloc(&_device_img, simg*sizeof(pixel_t)));
		picviz_verify_cuda(cudaMemset(_device_img, 0, simg*sizeof(pixel_t)));
		_org_width = width;
	}

public:
	virtual ~PVBCIBackendImageCUDA()
	{
		size_t simg = size_pixel();
		pixel_allocator().deallocate(_host_img, simg);
		picviz_verify_cuda(cudaFree(_device_img));
	}

protected:
	pixel_pointer_t device_img() { return _device_img; }
	pixel_pointer_t host_img() { return _host_img; }

	inline void copy_device_to_host(cudaStream_t stream) const
	{
		picviz_verify_cuda(cudaMemcpyAsync(_host_img, _device_img, size_pixel()*sizeof(pixel_t), cudaMemcpyDeviceToHost, stream));
	}

	inline void copy_device_to_host() const
	{
		picviz_verify_cuda(cudaMemcpy(_host_img, _device_img, size_pixel()*sizeof(pixel_t), cudaMemcpyDeviceToHost));
	}

public:
	virtual QImage qimage() const
	{
		// This is really important to use the const constructor version of QImage, so that no data copy is made !
		return QImage((const uchar*) _host_img, width(), PVParallelView::ImageHeight, QImage::Format_ARGB32_Premultiplied);
	}

	virtual bool set_width(uint32_t width)
	{
		if (width > _org_width) {
			return false;
		}

		return PVBCIBackendImage::set_width(width);
	}

private:
	pixel_pointer_t _host_img;
	pixel_pointer_t _device_img;
	uint32_t _org_width;
};

} }

PVParallelView::PVBCIDrawingBackendCUDA::PVBCIDrawingBackendCUDA()
{
	picviz_verify_cuda(cudaMalloc(&_device_codes, NBUCKETS*sizeof(PVBCICode)));
	picviz_verify_cuda(cudaStreamCreate(&_main_stream));
}

PVParallelView::PVBCIDrawingBackendCUDA::~PVBCIDrawingBackendCUDA()
{
	picviz_verify_cuda(cudaFree(_device_codes));
	picviz_verify_cuda(cudaStreamDestroy(_main_stream));
}

PVParallelView::PVBCIBackendImage_p PVParallelView::PVBCIDrawingBackendCUDA::create_image(size_t img_width) const
{
	return PVBCIBackendImage_p(new __impl::PVBCIBackendImageCUDA(img_width));
}

void PVParallelView::PVBCIDrawingBackendCUDA::operator()(PVBCIBackendImage& dst_img, size_t x_start, size_t width, PVBCICode* codes, size_t n) const
{
	__impl::PVBCIBackendImageCUDA* dst_img_cuda = dynamic_cast<__impl::PVBCIBackendImageCUDA*>(&dst_img);
	assert(dst_img_cuda != NULL);
	assert(x_start + width <= (size_t) dst_img_cuda->width());

	// Wait for that point in the stream
	cudaEvent_t end;
	cudaEventCreate(&end, 0);

#pragma omp critical
	{
		picviz_verify_cuda(cudaMemcpyAsync(_device_codes, codes, n*sizeof(codes), cudaMemcpyHostToDevice, _main_stream));
		show_codes_cuda(_device_codes, n, width, dst_img_cuda->device_img(), dst_img_cuda->width(), x_start, _main_stream);
		dst_img_cuda->copy_device_to_host(_main_stream);
		cudaEventRecord(end, _main_stream);
	}

	cudaEventSynchronize(end);
	cudaEventDestroy(end);
}
