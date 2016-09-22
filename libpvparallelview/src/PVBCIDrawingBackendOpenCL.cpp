/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

#include <pvkernel/core/PVLogger.h>
#include <pvkernel/core/PVConfig.h>

#include <pvkernel/opencl/common.h>
#include <pvkernel/core/PVHSVColor.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCIDrawingBackendOpenCL.h>
#include <pvparallelview/PVBCIBackendImageOpenCL.h>

#include <pvparallelview/PVBCICode.h>

#include <cassert>

#include <iostream>
#include <sstream>

#include <QSettings>

/* minimal required for OpenCL 1.0 devices
 */
#define LOCAL_MEMORY_SIZE (16 * 1024)

/******************************************************************************
 * opencl_kernel
 *****************************************************************************/

#include "bci_z24.h"

template <size_t Bbits>
struct opencl_kernel {
	static cl_int start(const PVParallelView::PVBCIDrawingBackendOpenCL::device_t& dev,
	                    cl::Kernel& kernel,
	                    const cl_uint n,
	                    const cl_uint width,
	                    const cl::Buffer& image_buffer,
	                    const cl_uint image_width,
	                    const cl_uint image_x_start,
	                    const cl_float zoom_y,
	                    const bool reverse)
	{
		const cl_uint bit_shift = Bbits;
		const cl_uint bit_mask = PVParallelView::constants<Bbits>::mask_int_ycoord;
		const cl_uint image_height = PVParallelView::constants<Bbits>::image_height;
		const size_t column_mem_size = image_height * sizeof(cl_uint);
		// bool is not a valid type as kernel parameter
		const cl_uint reverse_flag = reverse;

		inendi_verify_opencl(kernel.setArg(0, dev.buffer()));
		inendi_verify_opencl(kernel.setArg(1, n));
		inendi_verify_opencl(kernel.setArg(2, width));
		inendi_verify_opencl(kernel.setArg(3, image_buffer()));
		inendi_verify_opencl(kernel.setArg(4, image_width));
		inendi_verify_opencl(kernel.setArg(5, image_height));
		inendi_verify_opencl(kernel.setArg(6, image_x_start));
		inendi_verify_opencl(kernel.setArg(7, zoom_y));
		inendi_verify_opencl(kernel.setArg(8, bit_shift));
		inendi_verify_opencl(kernel.setArg(9, bit_mask));
		inendi_verify_opencl(kernel.setArg(10, reverse_flag));

		/* we make fit the highest number of image column in the work group local memory
		 */
		const size_t local_num_x = std::min((size_t)width, LOCAL_MEMORY_SIZE / column_mem_size);
		const size_t local_num_y = dev.work_group_size / local_num_x;

		const size_t global_num_x = ((width + local_num_x - 1) / local_num_x) * local_num_x;
		const size_t global_num_y = local_num_y;

		const cl::NDRange global_work(global_num_x, global_num_y);
		const cl::NDRange local_work(local_num_x, local_num_y);

		return dev.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work, local_work);
	}
};

/*****************************************************************************
 * PVParallelView::PVBCIDrawingBackendOpenCL::PVBCIDrawingBackendOpenCL
 *****************************************************************************/

PVParallelView::PVBCIDrawingBackendOpenCL::PVBCIDrawingBackendOpenCL()
    : _context(nullptr), _is_gpu_accelerated(true)
{
	size_t size = PVParallelView::MaxBciCodes * sizeof(PVBCICodeBase);
	int dev_idx = 0;
	cl_int err;

	auto& config = PVCore::PVConfig::get().config();
	bool force_cpu = config.value("backend_opencl/force_cpu", false).toBool();

	// List all usable OpenCL devices and create appropriate structures
	const auto fun = [&](cl::Context& ctx, cl::Device& dev) {
		device_t device;
		cl_int err;

		device.dev = dev;

		device.queue = cl::CommandQueue(ctx, dev, 0, &err);
		inendi_verify_opencl_var(err);

		device.buffer = cl::Buffer(ctx, CL_MEM_READ_ONLY, size, nullptr, &err);
		inendi_verify_opencl_var(err);

		this->_devices.insert(std::make_pair(dev_idx, device));
		++dev_idx;
	};

	if (force_cpu == false) {
		_context = PVOpenCL::find_first_usable_context(true, fun);
	}

	if (_context() == nullptr) {
		_context = PVOpenCL::find_first_usable_context(false, fun);
		_is_gpu_accelerated = false;
	}

	if (_context() == nullptr) {
		throw PVOpenCL::exception::no_backend_error();
	}

	_next_device = _devices.begin();

	cl::Program program(_context, bci_z24_str, false, &err);
	inendi_verify_opencl_var(err);

	/**
	 * NOTE: options can be passed to build process, like -DVAR=VAL. So that, Bbits
	 * dependant values and reverse can be passed at build time to decrease parameter
	 * count and have better optimisations.
	 */
	std::stringstream build_options;
	build_options << "-DLOCAL_MEMORY_SIZE=" << LOCAL_MEMORY_SIZE;
	build_options << " -DHSV_COLOR_COUNT=" << (int)PVCore::PVHSVColor::color_max;
	build_options << " -DHSV_COLOR_WHITE=" << (int)HSV_COLOR_WHITE.h();
	build_options << " -DHSV_COLOR_BLACK=" << (int)HSV_COLOR_BLACK.h();
	build_options << " -DHSV_COLOR_RED=" << (int)HSV_COLOR_RED.h();

	std::vector<cl::Device> devices = _context.getInfo<CL_CONTEXT_DEVICES>(&err);
	inendi_verify_opencl_var(err);

	err = program.build(devices, build_options.str().c_str());

#ifdef INENDI_DEVELOPER_MODE
	if (err != CL_SUCCESS) {
		/* As we build (implicitly) on all devices, we check every for errors
		 */
		for (const auto& dev : devices) {
			cl_build_status status;

			inendi_verify_opencl(program.getBuildInfo(dev, CL_PROGRAM_BUILD_STATUS, &status));

			if (status != CL_BUILD_ERROR) {
				continue;
			}

			std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);

			PVLOG_INFO("build log: %s\n", log.c_str());
		}
	}
#endif
	inendi_verify_opencl_var(err);

	_kernel = cl::Kernel(program, "DRAW", &err);
	inendi_verify_opencl_var(err);

	for (auto& it : _devices) {
		err = _kernel.getWorkGroupInfo(it.second.dev, CL_KERNEL_WORK_GROUP_SIZE,
		                               &it.second.work_group_size);
		inendi_verify_opencl_var(err);
	}
}

/*****************************************************************************
 * PVParallelView::PVBCIDrawingBackendOpenCL::get
 *****************************************************************************/

PVParallelView::PVBCIDrawingBackendOpenCL& PVParallelView::PVBCIDrawingBackendOpenCL::get()
{
	static PVBCIDrawingBackendOpenCL backend;
	return backend;
}

/*****************************************************************************
 * PVParallelView::PVBCIDrawingBackendOpenCL::create_image
 *****************************************************************************/

PVParallelView::PVBCIBackendImage_p
PVParallelView::PVBCIDrawingBackendOpenCL::create_image(size_t image_width, uint8_t height_bits)
{
	assert(_devices.size() >= 1);

	if (_next_device == _devices.end()) {
		_next_device = _devices.begin();
	}

	// Create image on a device in a round robin way
	const cl::CommandQueue& queue = _next_device->second.queue;

	PVBCIBackendImage_p ret(new PVBCIBackendImageOpenCL(image_width, height_bits, _context, queue,
	                                                    _next_device->first));

	++_next_device;

	return ret;
}

/*****************************************************************************
 * PVParallelView::PVBCIDrawingBackendOpenCL::allocate_bci
 *****************************************************************************/

PVParallelView::PVBCICodeBase* PVParallelView::PVBCIDrawingBackendOpenCL::allocate_bci(size_t n)
{
	return new PVBCICodeBase[n];
}

/*****************************************************************************
 * PVParallelView::PVBCIDrawingBackendOpenCL::free_bci
 *****************************************************************************/

void PVParallelView::PVBCIDrawingBackendOpenCL::free_bci(PVParallelView::PVBCICodeBase* buf)
{
	delete[] buf;
}

/*****************************************************************************
 * PVParallelView::PVBCIDrawingBackendOpenCL::operator()
 *****************************************************************************/

void PVParallelView::PVBCIDrawingBackendOpenCL::operator()(PVBCIBackendImage_p& backend_img,
                                                           size_t x_start,
                                                           size_t width,
                                                           PVBCICodeBase* codes,
                                                           size_t n,
                                                           const float zoom_y,
                                                           bool reverse,
                                                           std::function<void()> const& render_done)
{
#ifdef NDEBUG
	backend_image_t* dst_img = static_cast<backend_image_t*>(backend_img.get());
#else
	backend_image_t* dst_img = dynamic_cast<backend_image_t*>(backend_img.get());
	assert(dst_img != nullptr);
#endif
	device_t& dev = _devices[dst_img->index()];

	cl_int err;

	if (n != 0) {
		// Specs that a size of zero will lead to CL_INVALID_VALUE
		err = dev.queue.enqueueWriteBuffer(dev.buffer, CL_FALSE, 0, n * sizeof(codes), codes);
		inendi_verify_opencl_var(err);
	}

	switch (dst_img->height_bits()) {
	case 10:
		assert(reverse == false && "no reverse mode allowed in kernel<10>");

		err = opencl_kernel<10>::start(dev, _kernel, n, width, dst_img->device_buffer(),
		                               dst_img->width(), x_start, zoom_y, reverse);
		break;
	case 11:
		err = opencl_kernel<11>::start(dev, _kernel, n, width, dst_img->device_buffer(),
		                               dst_img->width(), x_start, zoom_y, reverse);
		break;
	default:
		assert(false);
		break;
	}
	inendi_verify_opencl_var(err);

	auto data = new opencl_job_data_t;
	data->done_function = render_done;

	dst_img->copy_device_to_host_async(&data->event);

	err = data->event.setCallback(CL_COMPLETE, &PVBCIDrawingBackendOpenCL::termination_cb, data);
	inendi_verify_opencl_var(err);

	// CPU drivers need to do an explicit clFlush to make event happen... strange...
	if (not _is_gpu_accelerated) {
		dev.queue.flush();
	}
}

/*****************************************************************************
 * PVParallelView::PVBCIDrawingBackendOpenCL::wait_all
 *****************************************************************************/

void PVParallelView::PVBCIDrawingBackendOpenCL::wait_all() const
{
	// Wait for all devices processing termination
	for (auto& device : _devices) {
		device.second.queue.finish();
	}
}

/*****************************************************************************
 * PVParallelView::PVBCIDrawingBackendOpenCL::termination_cb
 *****************************************************************************/

void PVParallelView::PVBCIDrawingBackendOpenCL::termination_cb(cl_event /* event */,
                                                               cl_int /* status */,
                                                               void* data)
{
	opencl_job_data_t* job_data = reinterpret_cast<opencl_job_data_t*>(data);

	// Call termination function
	if (job_data->done_function) {
		(job_data->done_function)();
	}

	delete job_data;
}
