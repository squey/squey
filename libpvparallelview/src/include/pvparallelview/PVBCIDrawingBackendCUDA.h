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

#include <QImage>
#include <cassert>

namespace PVParallelView {

class PVBCIDrawingBackendCUDA;

class PVBCIBackendImageCUDA: public PVParallelView::PVBCIBackendImage
{
	friend class PVParallelView::PVBCIDrawingBackendCUDA;
	friend class cuda_engine;
	
	typedef uint32_t pixel_t;
	typedef uint32_t* pixel_pointer_t;
	typedef uint32_t const* const_pixel_pointer_t;

	typedef std::allocator<pixel_t> pixel_allocator;

protected:
	PVBCIBackendImageCUDA(const uint32_t width, uint8_t height_bits, const int cuda_device);

public:
	virtual ~PVBCIBackendImageCUDA();

public:
	virtual void resize_width(PVBCIBackendImage& dst, const uint32_t width) const override;

protected:
	pixel_pointer_t device_img() { return _device_img; }
	pixel_pointer_t host_img() { return _host_img; }

	const_pixel_pointer_t device_img() const { return _device_img; }
	const_pixel_pointer_t host_img() const { return _host_img; }

	inline void copy_device_to_host(cudaStream_t const& stream) const
	{
		picviz_verify_cuda(cudaMemcpyAsync(_host_img, _device_img, size_org_pixel()*sizeof(pixel_t), cudaMemcpyDeviceToHost, stream));
	}

	inline void copy_device_to_host() const
	{
		picviz_verify_cuda(cudaMemcpy(_host_img, _device_img, size_org_pixel()*sizeof(pixel_t), cudaMemcpyDeviceToHost));
	}

	inline size_t org_width() const { return _org_width; }

	void set_current_device() const
	{
		picviz_verify_cuda(cudaSetDevice(_cuda_device));
	}

	int get_cuda_device() const { return _cuda_device; }

public:
	virtual QImage qimage() const
	{
		// This is really important to use the const constructor version of QImage, so that no data copy is made !
		return QImage((const uchar*) _host_img, PVBCIBackendImage::width(), PVBCIBackendImage::height(), _org_width*sizeof(uint32_t), QImage::Format_ARGB32_Premultiplied);
	}

	virtual bool set_width(uint32_t width)
	{
		if (width > _org_width) {
			return false;
		}

		return PVBCIBackendImage::set_width(width);
	}

private:
	inline size_t size_org_pixel() const { return _org_width*PVBCIBackendImage::height(); }

private:
	pixel_pointer_t _host_img;
	pixel_pointer_t _device_img;
	uint32_t _org_width;
	int _cuda_device;
};

class PVBCIDrawingBackendCUDA: public PVBCIDrawingBackendSync
{
	typedef PVBCIBackendImageCUDA backend_image_t;

	struct device_t
	{
		PVBCICodeBase* device_codes;
		cudaStream_t stream;
	};

private:
	typedef typename backend_image_t::pixel_t pixel_t;
	typedef typename backend_image_t::pixel_pointer_t pixel_pointer_t;

public:
	PVBCIDrawingBackendCUDA();
	virtual ~PVBCIDrawingBackendCUDA();

public:
	Flags flags() const { return Serial; }
	PVBCIBackendImage_p create_image(size_t img_width, uint8_t height_bits) const override;

public:
	void operator()(PVBCIBackendImage& dst_img, size_t x_start, size_t width, PVBCICodeBase* codes, size_t n, const float zoom_y = 1.0f, bool reverse = false, std::function<void()> const& render_done = std::function<void()>()) override;

private:
	mutable std::map<int, device_t> _devices;
	mutable std::map<int, device_t>::const_iterator _last_image_dev;
	cudaStream_t _main_stream;
};

}

#endif
