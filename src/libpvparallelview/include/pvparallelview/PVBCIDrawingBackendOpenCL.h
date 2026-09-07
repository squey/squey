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

#ifndef PVPARALLELVIEW_PVBCIDRAWINGBACKENDOPENCL_H
#define PVPARALLELVIEW_PVBCIDRAWINGBACKENDOPENCL_H

#include <pvkernel/opencl/common.h>

#include <pvparallelview/PVBCIDrawingBackend.h>

#include <functional>
#include <map>
#include <vector>

namespace PVParallelView
{

class PVBCIBackendImageOpenCL;

class PVBCIDrawingBackendOpenCL : public PVBCIDrawingBackendAsync
{
	using backend_image_t = PVBCIBackendImageOpenCL;

	struct opencl_job_data_t {
		cl::Event event;
		std::function<void()> done_function;
	};

  public:
	struct device_t {
		cl::Device dev;
		cl::Buffer buffer;
		cl::CommandQueue queue;
		size_t work_group_size;
		cl_ulong local_mem_size;
	};

  public:
	PVBCIDrawingBackendOpenCL();

  public:
	bool is_gpu_accelerated() const override { return _is_gpu_accelerated; }
	size_t device_count() const { return _devices.size(); }

	/**
	 * Compiles every kernel the views will ask for, so that none of them has to
	 * be compiled while a zone is waiting to be drawn.
	 *
	 * PortableCL specialises a kernel per local work size and only does so on
	 * the first enqueue naming one, which is why a zone width drawn for the
	 * first time used to stay black for as long as the compiler took. Drawing
	 * each shape once with nothing in it pays that cost here instead, and the
	 * kernel cache keeps it for the next runs.
	 *
	 * progress, if given, is called with (shapes done, shapes total) before
	 * each one and once more at the end.
	 */
	void precompile_kernels(const std::function<void(size_t, size_t)>& progress = {});

  public:
	static PVBCIDrawingBackendOpenCL& get();

  public:
	Flags flags() const override { return Serial; }

	PVBCIBackendImage_p create_image(size_t img_width, uint8_t height_bits) override;

  public:
	PVBCICodeBase* allocate_bci(size_t n) override;

	void free_bci(PVBCICodeBase* buf) override;

  public:
	void render(PVBCIBackendImage_p& dst_img,
	            size_t x_start,
	            size_t width,
	            PVBCICodeBase* codes,
	            size_t n,
	            const float zoom_y = 1.0f,
	            bool reverse = false,
	            std::function<void()> const& render_done = std::function<void()>()) override;

	void wait_all() const override;

  private:
	/**
	 * Callback function called once image creation is done and back on computer.
	 */
	static void termination_cb(cl_event event, cl_int status, void* data);

	auto create_new_image(backend_image_t* in_place, size_t img_width, uint8_t height_bits)
	    -> backend_image_t*;

  private:
#if defined __GNUC__ && __GNUC__ >= 6
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif
	using devices_t = std::map<cl_int, device_t>;
#if defined __GNUC__ && __GNUC__ >= 6
#pragma GCC diagnostic pop
#endif

  private:
	cl::Context _context;
	cl::Kernel _kernel;
	devices_t _devices;
	devices_t::const_iterator _next_device;
	bool _is_gpu_accelerated;
};

} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVBCIDRAWINGBACKENDOPENCL_H
