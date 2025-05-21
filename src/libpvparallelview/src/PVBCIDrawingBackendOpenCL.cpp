//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/core/PVLogger.h>
#include <pvkernel/core/PVConfig.h>
#include <pvkernel/core/PVUtils.h>

#include <pvkernel/opencl/common.h>
#include <pvkernel/core/PVHSVColor.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCIDrawingBackendOpenCL.h>
#include <pvparallelview/PVBCIBackendImageOpenCL.h>

#include <pvparallelview/PVBCICode.h>

#include <cassert>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <filesystem>

#include <QSettings>

#include <boost/dll/runtime_symbol_info.hpp>

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

		squey_verify_opencl(kernel.setArg(0, dev.buffer()));
		squey_verify_opencl(kernel.setArg(1, n));
		squey_verify_opencl(kernel.setArg(2, width));
		squey_verify_opencl(kernel.setArg(3, image_buffer()));
		squey_verify_opencl(kernel.setArg(4, image_width));
		squey_verify_opencl(kernel.setArg(5, image_height));
		squey_verify_opencl(kernel.setArg(6, image_x_start));
		squey_verify_opencl(kernel.setArg(7, zoom_y));
		squey_verify_opencl(kernel.setArg(8, bit_shift));
		squey_verify_opencl(kernel.setArg(9, bit_mask));
		squey_verify_opencl(kernel.setArg(10, reverse_flag));

		/* we make fit the highest number of image column in the work group local memory
		 */
		const size_t local_num_x =
		    std::min((cl_ulong)width, (dev.local_mem_size / column_mem_size) - 1);
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
	PVCore::setenv("POCL_CPU_LOCAL_MEM_SIZE", std::to_string(PARALLELVIEW_POCL_CPU_LOCAL_MEM_SIZE).c_str(), 0);

#ifdef __APPLE__
	// Configure our patched PortableCL to find "ld64.lld" linker at runtime
	boost::filesystem::path exe_path = boost::dll::program_location();
	PVCore::setenv("POCL_LINKER_DIR", exe_path.parent_path().string().c_str(), 1);
#elifdef _WIN32
	// Configure "ld" linker to search for librairies in the proper location
	// and Khronos ICD loader to find PortableCL
	boost::filesystem::path exe_path = boost::dll::program_location();
	std::string libdir = exe_path.parent_path().string();
	PVCore::setenv("LIBRARY_PATH", libdir.c_str(), 1);
	std::string pocl_path = libdir + "/pocl.dll";
	PVCore::setenv("OCL_ICD_FILENAMES", pocl_path.c_str(), 1);
	std::filesystem::current_path(libdir);
#endif

	size_t size = PVParallelView::MaxBciCodes * sizeof(PVBCICodeBase);
	int dev_idx = 0;
	cl_int err;
	const cl_uint Bbits = PARALLELVIEW_ZZT_BBITS;
	const cl_uint image_height = PVParallelView::constants<Bbits>::image_height;
	const size_t column_mem_size = image_height * sizeof(cl_uint);
	const uint64_t max_mem = column_mem_size * PARALLELVIEW_ZONE_MAX_WIDTH;

	auto& config = PVCore::PVConfig::get().config();
	bool force_cpu = config.value("backend_opencl/force_cpu", false).toBool();
	const char* force_cpu_env = getenv("FORCE_CPU");
	force_cpu |= (force_cpu_env != nullptr && std::string(force_cpu_env) == "1");

	// List all usable OpenCL devices and create appropriate structures
	const auto fun = [&](cl::Context& ctx, cl::Device& dev) {
		device_t device{};
		cl_int err;

		device.dev = dev;

		device.queue = cl::CommandQueue(ctx, dev, 0, &err);
		squey_verify_opencl_var(err);

		device.buffer = cl::Buffer(ctx, CL_MEM_READ_ONLY, size, nullptr, &err);
		squey_verify_opencl_var(err);

		this->_devices.insert(std::make_pair(dev_idx, device));
		++dev_idx;
	};

	if (force_cpu == false) {
		_context = PVOpenCL::find_first_usable_context(true, fun);
	}
	else {
		_is_gpu_accelerated = false;
	}

	if (_context() == nullptr) {
		_context = PVOpenCL::find_first_usable_context(false, fun);
		_is_gpu_accelerated = false;
	}

	if (_context() == nullptr) {
		PVLOG_INFO("No OpenCL support: no context available.\n");
		return;
	}

	_next_device = _devices.begin();

	cl::Program program(_context, bci_z24_str, false, &err);
	squey_verify_opencl_var(err);

	/**
	 * NOTE: options can be passed to build process, like -DVAR=VAL. So that, Bbits
	 * dependant values and reverse can be passed at build time to decrease parameter
	 * count and have better optimisations.
	 */

	std::vector<cl::Device> devices = _context.getInfo<CL_CONTEXT_DEVICES>(&err);
	squey_verify_opencl_var(err);

	uint64_t local_mem_size;
	for (auto& it : _devices) {
		err = it.second.dev.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &local_mem_size);
		squey_verify_opencl_var(err);
	}

	std::stringstream build_options;
	build_options << "-DLOCAL_MEMORY_SIZE=" << std::min(max_mem, (local_mem_size - 1));
	build_options << " -DHSV_COLOR_COUNT=" << (int)PVCore::PVHSVColor::color_max;
	build_options << " -DHSV_COLOR_WHITE=" << (int)HSV_COLOR_WHITE.h();
	build_options << " -DHSV_COLOR_BLACK=" << (int)HSV_COLOR_BLACK.h();
	build_options << " -DHSV_COLOR_RED=" << (int)HSV_COLOR_RED.h();

	err = program.build(devices, build_options.str().c_str());

	if (err != CL_SUCCESS) {
		/* As we build (implicitly) on all devices, we check every for errors
		 */
		for (const auto& dev : devices) {
			cl_build_status status;

			squey_verify_opencl(program.getBuildInfo(dev, CL_PROGRAM_BUILD_STATUS, &status));

			if (status != CL_BUILD_ERROR) {
				continue;
			}

			std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
			PVLOG_INFO("build log: %s\n", log.c_str());
		}
	}
	squey_verify_opencl_var(err);

	_kernel = cl::Kernel(program, "DRAW", &err);
	squey_verify_opencl_var(err);

	for (auto& it : _devices) {
		err = _kernel.getWorkGroupInfo(it.second.dev, CL_KERNEL_WORK_GROUP_SIZE,
		                               &it.second.work_group_size);
		squey_verify_opencl_var(err);

		err = it.second.dev.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &it.second.local_mem_size);
		squey_verify_opencl_var(err);
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
PVParallelView::PVBCIDrawingBackendOpenCL::create_image(size_t /*image_width*/, uint8_t height_bits)
{
	// The minimal possible > 0, this is a workaround (an ugly one)
	return PVBCIBackendImage_p(create_new_image(nullptr, 2, height_bits));
}

auto PVParallelView::PVBCIDrawingBackendOpenCL::create_new_image(backend_image_t* in_place,
                                                                 size_t image_width,
                                                                 uint8_t height_bits)
    -> backend_image_t*
{
	assert(_devices.size() >= 1);

	if (_next_device == _devices.end()) {
		_next_device = _devices.begin();
	}

	// Create image on a device in a round robin way
	const cl::CommandQueue& queue = _next_device->second.queue;

	if (in_place == nullptr) {
		in_place = new PVBCIBackendImageOpenCL(image_width, height_bits, _context, queue,
		                                       _next_device->first);
	} else {
		in_place->~PVBCIBackendImageOpenCL();
		new (in_place)
		    PVBCIBackendImageOpenCL(image_width, height_bits, _context, queue, _next_device->first);
	}

	if (_devices.size() > 1) {
		++_next_device;
	}

	return in_place;
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
 * PVParallelView::PVBCIDrawingBackendOpenCL::render
 *****************************************************************************/

void PVParallelView::PVBCIDrawingBackendOpenCL::render(PVBCIBackendImage_p& backend_img,
                                                       size_t x_start,
                                                       size_t width,
                                                       PVBCICodeBase* codes,
                                                       size_t n,
                                                       const float zoom_y,
                                                       bool reverse,
                                                       std::function<void()> const& render_done)
{
#ifdef NDEBUG
	auto* dst_img = static_cast<backend_image_t*>(backend_img.get());
#else
	backend_image_t* dst_img = dynamic_cast<backend_image_t*>(backend_img.get());
	assert(dst_img != nullptr);
#endif

	if (dst_img->width() != width) {
		auto height_bits = dst_img->height_bits();
		create_new_image(dst_img, width, height_bits);
	}

	device_t& dev = _devices[dst_img->index()];

	cl_int err;

	if (n != 0) {
		// Specs that a size of zero will lead to CL_INVALID_VALUE
		err = dev.queue.enqueueWriteBuffer(dev.buffer, CL_FALSE, 0, n * sizeof(codes), codes);
		squey_verify_opencl_var(err);
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
	squey_verify_opencl_var(err);

	auto data = new opencl_job_data_t;
	data->done_function = render_done;

	dst_img->copy_device_to_host_async(&data->event);

	err = data->event.setCallback(CL_COMPLETE, &PVBCIDrawingBackendOpenCL::termination_cb, data);
	squey_verify_opencl_var(err);

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
	auto* job_data = reinterpret_cast<opencl_job_data_t*>(data);

	// Call termination function
	if (job_data->done_function) {
		(job_data->done_function)();
	}

	delete job_data;
}
