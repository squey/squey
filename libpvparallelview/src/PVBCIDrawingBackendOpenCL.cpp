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
	                    const cl_kernel kernel,
	                    const cl_uint n,
	                    const cl_uint width,
	                    const cl_mem image_mem,
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

		inendi_verify_opencl(clSetKernelArg(kernel, 0, sizeof(cl_mem), &dev.mem));
		inendi_verify_opencl(clSetKernelArg(kernel, 1, sizeof(cl_uint), &n));
		inendi_verify_opencl(clSetKernelArg(kernel, 2, sizeof(cl_uint), &width));
		inendi_verify_opencl(clSetKernelArg(kernel, 3, sizeof(cl_mem), &image_mem));
		inendi_verify_opencl(clSetKernelArg(kernel, 4, sizeof(cl_uint), &image_width));
		inendi_verify_opencl(clSetKernelArg(kernel, 5, sizeof(cl_uint), &image_height));
		inendi_verify_opencl(clSetKernelArg(kernel, 6, sizeof(cl_uint), &image_x_start));
		inendi_verify_opencl(clSetKernelArg(kernel, 7, sizeof(cl_float), &zoom_y));
		inendi_verify_opencl(clSetKernelArg(kernel, 8, sizeof(cl_uint), &bit_shift));
		inendi_verify_opencl(clSetKernelArg(kernel, 9, sizeof(cl_uint), &bit_mask));
		inendi_verify_opencl(clSetKernelArg(kernel, 10, sizeof(cl_uint), &reverse_flag));

		/* we make fit the highest number of image column in the work group local memory
		 */
		const size_t local_num_x = std::min((size_t)width, LOCAL_MEMORY_SIZE / column_mem_size);
		const size_t local_num_y = dev.work_group_size / local_num_x;

		const size_t global_num_x = ((width + local_num_x - 1) / local_num_x) * local_num_x;
		const size_t global_num_y = local_num_y;

		const size_t global_work[] = {global_num_x, global_num_y};
		const size_t local_work[] = {local_num_x, local_num_y};

		return clEnqueueNDRangeKernel(dev.queue, kernel, 2, nullptr, global_work, local_work, 0,
		                              nullptr, nullptr);
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
	const auto fun = [&](cl_context ctx, cl_device_id dev_id) {
		device_t dev;
		cl_int err;

		dev.id = dev_id;

		dev.queue = clCreateCommandQueue(ctx, dev_id, 0, &err);
		inendi_verify_opencl_var(err);

		dev.addr = PVOpenCL::host_allocate<PVBCICodeBase>(ctx, dev.queue, CL_MEM_READ_ONLY,
		                                                  CL_MAP_WRITE, size, dev.mem, err);
		inendi_verify_opencl_var(err);
		assert(dev.addr != nullptr);

		this->_devices.insert(std::make_pair(dev_idx, dev));
		++dev_idx;
	};

	if (force_cpu == false) {
		_context = PVOpenCL::find_first_usable_context(true, fun);
	}

	if (_context == nullptr) {
		_context = PVOpenCL::find_first_usable_context(false, fun);
		_is_gpu_accelerated = false;
	}

	if (_context == nullptr) {
		throw PVOpenCL::exception::no_backend_error();
	}

	_next_device = _devices.begin();

	/* NOTE: "1" because the kernel code is stored using an array of nul-terminated string.
	 */
	cl_program program = clCreateProgramWithSource(_context, 1, &bci_z24_str, nullptr, &err);
	inendi_verify_opencl_var(err);

	/**
	 * NOTE: options can be passed to build process, like -DVAR=VAL. So that, Bbits
	 * dependant values and reverse can be passed at build time to decrease parameter
	 * count and have better optimisations.
	 */
	std::stringstream build_options;
	build_options << "-DLOCAL_MEMORY_SIZE=" << LOCAL_MEMORY_SIZE;
	build_options << " -DHSV_COLOR_COUNT=" << HSV_COLOR_COUNT;
	build_options << " -DHSV_COLOR_WHITE=" << HSV_COLOR_WHITE;
	build_options << " -DHSV_COLOR_BLACK=" << HSV_COLOR_BLACK;
	build_options << " -DHSV_COLOR_RED=" << HSV_COLOR_RED;

	err = clBuildProgram(program, 0, nullptr, build_options.str().c_str(), nullptr, nullptr);

#ifdef INENDI_DEVELOPER_MODE
	if (err != CL_SUCCESS) {
		/* As we build (implicitly) on all devices, we check every for errors
		 */
		for (auto& it : _devices) {
			cl_build_status status;

			clGetProgramBuildInfo(program, it.second.id, CL_PROGRAM_BUILD_STATUS, sizeof(status),
			                      &status, nullptr);

			if (status != CL_BUILD_ERROR) {
				continue;
			}

			size_t len;
			char buffer[256];
			clGetProgramBuildInfo(program, it.second.id, CL_PROGRAM_BUILD_LOG, sizeof(buffer),
			                      buffer, &len);
			PVLOG_INFO("build log: %s\n", buffer);
		}
	}
#endif
	inendi_verify_opencl_var(err);

	_kernel = clCreateKernel(program, "DRAW", &err);
	inendi_verify_opencl_var(err);

	for (auto& it : _devices) {
		err = clGetKernelWorkGroupInfo(_kernel, it.second.id, CL_KERNEL_WORK_GROUP_SIZE,
		                               sizeof(size_t), &it.second.work_group_size, nullptr);
		inendi_verify_opencl_var(err);
	}

	err = clReleaseProgram(program);
	inendi_verify_opencl_var(err);
}

/*****************************************************************************
 * PVParallelView::PVBCIDrawingBackendOpenCL::~PVBCIDrawingBackendOpenCL
 *****************************************************************************/

PVParallelView::PVBCIDrawingBackendOpenCL::~PVBCIDrawingBackendOpenCL()
{
	for (auto& dev : _devices) {
		inendi_verify_opencl_var(
		    PVOpenCL::host_free(dev.second.queue, dev.second.mem, dev.second.addr));
		inendi_verify_opencl(clReleaseCommandQueue(dev.second.queue));
	}

	clReleaseContext(_context);
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
	cl_command_queue queue = _next_device->second.queue;

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
	const size_t size = n * sizeof(PVParallelView::PVBCICodeBase);

	device_t dev;
	cl_int err;

	dev.queue = _devices[0].queue;
	dev.addr = PVOpenCL::host_allocate<PVBCICodeBase>(_context, dev.queue, CL_MEM_READ_WRITE,
	                                                  CL_MAP_READ, size, dev.mem, err);
	inendi_verify_opencl_var(err);
	assert(dev.addr != nullptr);

	_mapped_buffers.insert(std::make_pair(dev.addr, dev));

	return dev.addr;
}

/*****************************************************************************
 * PVParallelView::PVBCIDrawingBackendOpenCL::free_bci
 *****************************************************************************/

void PVParallelView::PVBCIDrawingBackendOpenCL::free_bci(PVParallelView::PVBCICodeBase* buf)
{
	auto entry = _mapped_buffers.find(buf);

	if (entry == _mapped_buffers.end()) {
		assert(false);
		return;
	}

	cl_int err = PVOpenCL::host_free(entry->second.queue, entry->second.mem, entry->second.addr);
	inendi_verify_opencl_var(err);

	_mapped_buffers.erase(buf);
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
	assert(dst_img != NULL);
#endif
	device_t& dev = _devices[dst_img->index()];

	cl_int err;

	if (n != 0) {
		// Specs that a size of zero will lead to CL_INVALID_VALUE
		err = clEnqueueWriteBuffer(dev.queue, dev.mem, CL_FALSE, 0, n * sizeof(codes), codes, 0,
		                           nullptr, nullptr);
		inendi_verify_opencl_var(err);
	}

	switch (dst_img->height_bits()) {
	case 10:
		assert(reverse == false && "no reverse mode allowed in kernel<10>");

		err = opencl_kernel<10>::start(dev, _kernel, n, width, dst_img->device_mem(),
		                               dst_img->width(), x_start, zoom_y, reverse);
		break;
	case 11:
		err = opencl_kernel<11>::start(dev, _kernel, n, width, dst_img->device_mem(),
		                               dst_img->width(), x_start, zoom_y, reverse);
		break;
	default:
		assert(false);
		break;
	};
	inendi_verify_opencl_var(err);

	opencl_job_data_t* data = new opencl_job_data_t;
	data->done_function = render_done;

	dst_img->copy_device_to_host_async(&data->event);

	err = clSetEventCallback(data->event, CL_COMPLETE, &PVBCIDrawingBackendOpenCL::termination_cb,
	                         data);
	inendi_verify_opencl_var(err);

	// CPU drivers need to do an explicit clFlush to make event happen... strange...
	if (not _is_gpu_accelerated) {
		clFlush(dev.queue);
	}
}

/*****************************************************************************
 * PVParallelView::PVBCIDrawingBackendOpenCL::wait_all
 *****************************************************************************/

void PVParallelView::PVBCIDrawingBackendOpenCL::wait_all() const
{
	// Wait for all devices processing termination
	for (auto& device : _devices) {
		clFinish(device.second.queue);
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

	clReleaseEvent(job_data->event);

	delete job_data;
}
