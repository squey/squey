/**
 * \file PVBCIDrawingBackendCUDA.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_PVBCIDRAWINGBACKENDCUDA_H
#define PVPARALLELVIEW_PVBCIDRAWINGBACKENDCUDA_H

#include <pvkernel/cuda/common.h>

#include <pvparallelview/PVBCIBackendImage.h>
#include <pvparallelview/PVBCIDrawingBackend.h>
#include <pvparallelview/cuda/bci_cuda.h>

#include <QImage>
#include <cassert>

namespace PVParallelView {

template <size_t Bbits>
class PVBCIDrawingBackendCUDA;

namespace __impl {

template <size_t Bbits = NBITS_INDEX>
class PVBCIBackendImageCUDA: public PVParallelView::PVBCIBackendImage<Bbits>
{
	friend class PVParallelView::PVBCIDrawingBackendCUDA<Bbits>;
	
	typedef uint32_t pixel_t;
	typedef uint32_t* pixel_pointer_t;

	typedef std::allocator<pixel_t> pixel_allocator;

protected:
	PVBCIBackendImageCUDA(const uint32_t width):
		PVBCIBackendImage<Bbits>(width)
	{
		size_t simg = PVBCIBackendImage<Bbits>::size_pixel();
		_host_img = pixel_allocator().allocate(simg);
		picviz_verify_cuda(cudaMalloc(&_device_img, simg*sizeof(pixel_t)));
		picviz_verify_cuda(cudaMemset(_device_img, 0, simg*sizeof(pixel_t)));
		_org_width = width;
	}

public:
	virtual ~PVBCIBackendImageCUDA()
	{
		size_t simg = PVBCIBackendImage<Bbits>::size_pixel();
		pixel_allocator().deallocate(_host_img, simg);
		picviz_verify_cuda(cudaFree(_device_img));
	}

protected:
	pixel_pointer_t device_img() { return _device_img; }
	pixel_pointer_t host_img() { return _host_img; }

	inline void copy_device_to_host(cudaStream_t stream) const
	{
		picviz_verify_cuda(cudaMemcpyAsync(_host_img, _device_img, PVBCIBackendImage<Bbits>::size_pixel()*sizeof(pixel_t), cudaMemcpyDeviceToHost, stream));
	}

	inline void copy_device_to_host() const
	{
		picviz_verify_cuda(cudaMemcpy(_host_img, _device_img, PVBCIBackendImage<Bbits>::size_pixel()*sizeof(pixel_t), cudaMemcpyDeviceToHost));
	}

public:
	virtual QImage qimage() const
	{
		// This is really important to use the const constructor version of QImage, so that no data copy is made !
		return QImage((const uchar*) _host_img, PVBCIBackendImage<Bbits>::width(), PVBCIBackendImage<Bbits>::height(), QImage::Format_ARGB32_Premultiplied);
	}

	virtual bool set_width(uint32_t width)
	{
		if (width > _org_width) {
			return false;
		}

		return PVBCIBackendImage<Bbits>::set_width(width);
	}

private:
	pixel_pointer_t _host_img;
	pixel_pointer_t _device_img;
	uint32_t _org_width;
};

template <size_t Bbits>
struct cuda_kernel;

template <>
struct cuda_kernel<10>
{
	static inline void launch(PVParallelView::PVBCICode<10>* device_codes, uint32_t n, uint32_t width, uint32_t* device_img, uint32_t img_width, uint32_t x_start, const float zoom_y, cudaStream_t stream = NULL, bool reverse = false)
	{
		assert(reverse == false);
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

}

template <size_t Bbits = NBITS_INDEX>
class PVBCIDrawingBackendCUDA: public PVBCIDrawingBackend<Bbits>
{
	constexpr static size_t image_height = ((size_t)1)<<Bbits;

	typedef PVParallelView::__impl::PVBCIBackendImageCUDA<Bbits> backend_image_t;

private:
	typedef typename backend_image_t::pixel_t pixel_t;
	typedef typename backend_image_t::pixel_pointer_t pixel_pointer_t;

public:
	PVBCIDrawingBackendCUDA()
	{
		picviz_verify_cuda(cudaMalloc(&_device_codes, NBUCKETS*sizeof(PVBCICode<Bbits>)));
		picviz_verify_cuda(cudaStreamCreate(&_main_stream));
	}
	virtual ~PVBCIDrawingBackendCUDA()
	{
		picviz_verify_cuda(cudaFree(_device_codes));
		picviz_verify_cuda(cudaStreamDestroy(_main_stream));
	}

public:
	PVBCIBackendImage_p<Bbits> create_image(size_t img_width) const
	{
		return PVBCIBackendImage_p<Bbits>(new backend_image_t(img_width));
	}

	void operator()(PVBCIBackendImage<Bbits>& dst_img, size_t x_start, size_t width, PVBCICode<Bbits>* codes, size_t n, const float zoom_y = 1.0f, bool reverse = false) const
	{
		backend_image_t* dst_img_cuda = dynamic_cast<backend_image_t*>(&dst_img);
		assert(dst_img_cuda != NULL);
		assert(x_start + width <= (size_t) dst_img_cuda->width());

		// Wait for that point in the stream
		cudaEvent_t end;
		cudaEventCreate(&end, 0);

#pragma omp critical
		{
			picviz_verify_cuda(cudaMemcpyAsync(_device_codes, codes, n*sizeof(codes), cudaMemcpyHostToDevice, _main_stream));
			__impl::cuda_kernel<Bbits>::launch(_device_codes, n, width, dst_img_cuda->device_img(), dst_img_cuda->width(), x_start, zoom_y, _main_stream, reverse);
			dst_img_cuda->copy_device_to_host(_main_stream);
			cudaEventRecord(end, _main_stream);
		}

		cudaEventSynchronize(end);
		cudaEventDestroy(end);
	}

private:
	// TODO: make that one per CUDA device !
	PVBCICode<Bbits>* _device_codes;
	cudaStream_t _main_stream;
};

}

#endif
