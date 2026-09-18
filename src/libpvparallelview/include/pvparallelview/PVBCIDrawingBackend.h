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

#ifndef PVPARALLELVIEW_PVBCIDRAWINGBACKEND_H
#define PVPARALLELVIEW_PVBCIDRAWINGBACKEND_H

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVBCIBackendImage_types.h>

#include <functional>
#include <string>
#include <vector>

namespace PVParallelView
{

/**
 * Management class to create image representation and fill it.
 */
class PVBCIDrawingBackend
{
  public:
	using backend_image_t = PVBCIBackendImage;
	using backend_image_p_t = PVBCIBackendImage_p;

	typedef enum { Serial = 1, Parallel = 2 } Flags;

	/**
	 * What an OpenCL device a backend draws on says of itself.
	 */
	struct opencl_device_t {
		std::string name;
		std::string vendor;
		std::string driver_version;
		std::string opencl_version;
	};

  public:
	virtual ~PVBCIDrawingBackend() = default;

  public:
	virtual bool is_gpu_accelerated() const = 0;

	/**
	 * @return the OpenCL devices this backend draws on; none for a backend that
	 * does not draw through OpenCL.
	 *
	 * Whoever reports what the views are drawn on asks here rather than
	 * searching for devices on its own: the OpenCL backend may well have found a
	 * GPU and then drawn on the CPU.
	 */
	virtual std::vector<opencl_device_t> opencl_devices() const { return {}; }

  public:
	virtual backend_image_p_t create_image(size_t img_width, uint8_t height_bits) = 0;
	// TODO : flags is only Serial.
	virtual Flags flags() const = 0;
	virtual bool is_sync() const = 0;

  public:
	virtual PVBCICodeBase* allocate_bci(size_t n)
	{
		return (PVBCICodeBase*)PVBCICode<>::allocate_codes(n);
	}
	virtual void free_bci(PVBCICodeBase* buf) { return PVBCICode<>::free_codes((PVBCICode<>*)buf); }

  public:
	// If this backend is synchronous, render_done must be ignored.
	virtual void render(PVBCIBackendImage_p& dst_img,
	                    size_t x_start,
	                    size_t width,
	                    PVBCICodeBase* codes,
	                    size_t n,
	                    const float zoom_y = 1.0f,
	                    bool reverse = false,
	                    std::function<void()> const& render_done = std::function<void()>()) = 0;
};

/**
 * Interface for asynchronous Drawing Backend.
 */
class PVBCIDrawingBackendAsync : public PVBCIDrawingBackend
{
  public:
	bool is_sync() const override { return false; }

  public:
	// TODO : Remove this unused method.
	virtual void wait_all() const = 0;
};
} // namespace PVParallelView

#endif
