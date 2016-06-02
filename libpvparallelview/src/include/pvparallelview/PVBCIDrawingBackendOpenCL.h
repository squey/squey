/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

#ifndef PVPARALLELVIEW_PVBCIDRAWINGBACKENDOPENCL_H
#define PVPARALLELVIEW_PVBCIDRAWINGBACKENDOPENCL_H

#include <pvkernel/opencl/common.h>

#include <pvparallelview/PVBCIDrawingBackend.h>

#include <map>
#include <vector>

namespace PVParallelView
{

class PVBCIBackendImageOpenCL;

class PVBCIDrawingBackendOpenCL : public PVBCIDrawingBackendAsync
{
	using backend_image_t = PVBCIBackendImageOpenCL;

	struct opencl_job_data_t {
		cl_event event;
		std::function<void()> done_function;
	};

  public:
	struct device_t {
		cl_device_id id;
		PVBCICodeBase* addr;
		cl_mem mem;
		cl_command_queue queue;
		size_t work_group_size;
	};

  public:
	PVBCIDrawingBackendOpenCL();

	virtual ~PVBCIDrawingBackendOpenCL();

  public:
	static PVBCIDrawingBackendOpenCL& get();

  public:
	Flags flags() const { return Serial; }

	PVBCIBackendImage_p create_image(size_t img_width, uint8_t height_bits) override;

  public:
	PVBCICodeBase* allocate_bci(size_t n) override;

	void free_bci(PVBCICodeBase* buf) override;

  public:
	void operator()(PVBCIBackendImage_p& dst_img,
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

  private:
	using devices_t = std::map<cl_int, device_t>;
	using host_buffers_t = std::map<PVBCICodeBase*, device_t>;

  private:
	cl_context _context;
	cl_kernel _kernel;
	devices_t _devices;
	devices_t::const_iterator _next_device;
	host_buffers_t _mapped_buffers;
	bool _is_gpu_accelerated;
};

} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVBCIDRAWINGBACKENDOPENCL_H
