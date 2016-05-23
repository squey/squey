/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

#ifndef PVPARALLELVIEW_PVBCIBACKENDIMAGEOPENCL_H
#define PVPARALLELVIEW_PVBCIBACKENDIMAGEOPENCL_H

#include <pvparallelview/PVBCIBackendImage.h>

#include <CL/cl.h>

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
	                        uint8_t height_bits,
	                        const cl_context context,
	                        const cl_command_queue queue,
	                        int index);

	~PVBCIBackendImageOpenCL();

  public:
	inline cl_command_queue queue() const { return _queue; }

	inline int index() const { return _index; }

	inline size_t width() const { return _width; }

  public:
	const pixel_t* host_img() const { return _host_addr; }

	cl_mem host_mem() const { return _host_mem; }

	cl_mem device_mem() const { return _device_mem; }

	void copy_device_to_host_async(cl_event* event = nullptr) const;

	void copy_device_to_host_sync() const;

  public:
	virtual QImage qimage(size_t crop_height) const;

	virtual bool set_width(uint32_t width)
	{
		if (width > _width) {
			return false;
		}

		return PVBCIBackendImage::set_width(width);
	}

  private:
	inline size_t size_pixel() const { return _width * PVBCIBackendImage::height(); }

  private:
	pixel_t* _host_addr;
	cl_mem _host_mem;
	cl_mem _device_mem;
	uint32_t _width;
	cl_command_queue _queue;
	int _index;
};

} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVBCIBACKENDIMAGEOPENCL_H
