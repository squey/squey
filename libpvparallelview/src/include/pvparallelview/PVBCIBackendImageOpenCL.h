/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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
