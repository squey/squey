/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

#ifndef PVPARALLELVIEW_PVBCIBACKENDIMAGEOPENCL_H
#define PVPARALLELVIEW_PVBCIBACKENDIMAGEOPENCL_H

#include <pvparallelview/PVBCIBackendImage.h>

#include <CL/cl.hpp>

#include <QImage>

namespace PVParallelView
{

class PVBCIDrawingBackendOpenCL;

/**
 * It is an image convertible to QImage with host/device management.
 *
 * Calls for data transfert have to be done manualy to make sure they are done
 * the right time
 */
class PVBCIBackendImageOpenCL : public PVParallelView::PVBCIBackendImage
{
	friend class PVParallelView::PVBCIDrawingBackendOpenCL;

	using pixel_t = uint32_t;

  public:
	PVBCIBackendImageOpenCL(const uint32_t width,
	                        const uint8_t height_bits,
	                        const cl::Context& context,
	                        const cl::CommandQueue& queue,
	                        int index);

	PVBCIBackendImageOpenCL(const PVBCIBackendImageOpenCL&) = delete;
	PVBCIBackendImageOpenCL(PVBCIBackendImageOpenCL&&) = delete;

	~PVBCIBackendImageOpenCL() override;

  public:
	inline int index() const { return _index; }

	inline size_t width() const { return _width; }

  public:
	const pixel_t* host_img() const { return _host_addr; }

	const cl::Buffer& host_buffer() const { return _host_buffer; }

	const cl::Buffer& device_buffer() const { return _device_buffer; }

	void copy_device_to_host_async(cl::Event* event = nullptr) const;

  public:
	QImage qimage(size_t crop_height) const override;

	bool set_width(uint32_t width) override
	{
		if (width > _width) {
			return false;
		}

		return PVBCIBackendImage::set_width(width);
	}

  private:
	inline size_t size_pixel() const { return _width * PVBCIBackendImage::height(); }

  private:
	pixel_t* _host_addr = nullptr;
	cl::Buffer _host_buffer;
	cl::Buffer _device_buffer;
	uint32_t _width;
	cl::CommandQueue _queue;
	int _index;
};

} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVBCIBACKENDIMAGEOPENCL_H
