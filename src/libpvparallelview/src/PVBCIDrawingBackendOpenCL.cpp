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
#include <pvkernel/core/PVUtils.h>

#include <pvkernel/opencl/common.h>
#include <pvkernel/core/PVHSVColor.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCIDrawingBackendOpenCL.h>
#include <pvparallelview/PVBCIBackendImageOpenCL.h>

#include <pvparallelview/PVBCICode.h>

#include <algorithm>
#include <cassert>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <filesystem>

#ifdef _WIN32
#include <windows.h>
#endif

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

		/* We make fit the highest number of image columns in the work group local
		 * memory. The shape must not follow the zone width, though: PortableCL
		 * specialises the kernel per local work size and caches the result, so a
		 * width never drawn before costs a compiler run -- and, where that run is
		 * slow, leaves the zone black until it ends. Rounding up to a power of two
		 * bounds the number of distinct shapes to a handful, at the price of the
		 * work-items past the zone that the kernel now lets through without
		 * drawing.
		 */
		const cl_ulong max_local_num_x =
		    std::min({(cl_ulong)PARALLELVIEW_ZONE_MAX_WIDTH,
		              (cl_ulong)dev.work_group_size,
		              (dev.local_mem_size / column_mem_size) - 1});
		size_t local_num_x = 1;
		while (local_num_x < width && (local_num_x * 2) <= max_local_num_x) {
			local_num_x *= 2;
		}
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
	// Configure our patched PortableCL to find "ld64.lld" linker at runtime.
	// It ships beside the application, which is where the executable is -- but
	// not for the test binaries: those run from the build tree while the linker
	// stays in the bundle, and pocl was handed a -fuse-ld= naming a file that
	// does not exist. Fall back to the PATH, which is how the test environment
	// reaches the bundle.
	boost::filesystem::path exe_path = boost::dll::program_location();
	boost::filesystem::path linker_dir = exe_path.parent_path();
	if (not std::filesystem::exists((linker_dir / "ld64.lld").string())) {
		const char* const path_env = std::getenv("PATH");
		std::istringstream path_stream(path_env != nullptr ? path_env : "");
		std::string dir;
		while (std::getline(path_stream, dir, ':')) {
			if (not dir.empty() and std::filesystem::exists(std::filesystem::path(dir) / "ld64.lld")) {
				linker_dir = dir;
				break;
			}
		}
	}
	PVCore::setenv("POCL_LINKER_DIR", linker_dir.string().c_str(), 1);
#elifdef _WIN32
	// Configure "ld" linker to search for librairies in the proper location
	// and Khronos ICD loader to find PortableCL
	boost::filesystem::path exe_path = boost::dll::program_location();
	std::string libdir = exe_path.parent_path().string();

	// pocl links each kernel it compiles against the mingw runtime, and finds
	// those through LIBRARY_PATH. They sit next to the application, which is
	// where the executable is -- except for the test binaries, installed two
	// levels below under tests/ (see CMakeMacros.txt). Pointing at their own
	// directory leaves ld with no libmingw32.a, no dllcrt2.o and no kernel:
	// every zone comes back blank. Walk up to whichever directory actually
	// holds them, and keep the executable's own as the last resort so that a
	// layout not anticipated here behaves as before.
	for (boost::filesystem::path dir = exe_path.parent_path(); not dir.empty();
	     dir = dir.parent_path()) {
		if (std::filesystem::exists((dir / "libmingw32.a").string())) {
			libdir = dir.string();
			break;
		}

		if (dir == dir.parent_path()) {
			break;
		}
	}

	PVCore::setenv("LIBRARY_PATH", libdir.c_str(), 1);
	// Beside the executable is where the packaged application finds it, squey.exe
	// and pocl.dll sitting in the same directory. The test executables are
	// installed two levels below that, under tests/ (see CMakeMacros.txt), and
	// naming a path that does not exist leaves the loader with no ICD at all --
	// which is why the testsuite ran without an OpenCL device. The bare name
	// falls back on the regular DLL search order, and the last argument of
	// setenv leaves an OCL_ICD_FILENAMES set by the caller alone.
	std::string pocl_path = libdir + "/pocl.dll";
	if (not std::filesystem::exists(pocl_path)) {
		pocl_path = "pocl.dll";
	}
	PVCore::setenv("OCL_ICD_FILENAMES", pocl_path.c_str(), 0);

	// The Khronos loader also reads this environment variable through
	// secure_getenv(), which Windows makes return NULL for a process running at
	// a high integrity level -- a deliberate hardening against an elevated
	// process picking up an attacker-controlled driver path from its
	// environment. GitLab's Windows CI runner executes tests at exactly that
	// level, which left every OpenCL-backed test with no device at all,
	// regardless of pocl.dll being perfectly correct and OCL_ICD_FILENAMES
	// pointing right at it: the loader never even looked.
	//
	// Registering the same path under the registry key the loader also reads
	// sidesteps the guard entirely, as it is a plain read with no secure_getenv
	// involved. A normal, non-elevated desktop session cannot write HKLM, so
	// this is additional to the environment variable above, not a replacement
	// for it: whichever one the current process is allowed to use is the one
	// that ends up mattering.
	HKEY icd_vendors_key;
	if (RegCreateKeyExA(HKEY_LOCAL_MACHINE, "SOFTWARE\\Khronos\\OpenCL\\Vendors", 0,
	                     nullptr, 0, KEY_SET_VALUE, nullptr, &icd_vendors_key,
	                     nullptr) == ERROR_SUCCESS) {
		DWORD icd_version = 0; // the ICD spec's "OpenCL 1.2 or later" marker
		RegSetValueExA(icd_vendors_key, pocl_path.c_str(), 0, REG_DWORD,
		               reinterpret_cast<const BYTE*>(&icd_version), sizeof(icd_version));
		RegCloseKey(icd_vendors_key);
	}

	// Where the loader is to look for pocl.dll and the DLLs it depends on. This
	// was a chdir, harmless while libdir was the directory the executable runs
	// from -- the tests already ran there. Now that it is wherever the mingw
	// runtime lives, two levels above them, moving there would resolve every
	// relative path the process opens afterwards from the wrong place: the test
	// files, named relative to the test's own directory, stopped being found.
	// This adds the directory to the DLL search order and leaves the working
	// directory alone.
	SetDllDirectoryA(libdir.c_str());

	// The chdir was also how ld found the startup files, which the clang driver
	// names without a path and ld then only looks for in the working directory.
	// Name the directory instead: pocl's linker flags carry a keyword it swaps
	// for this at link time (-B, see portablecl.bst), so the driver resolves
	// dllcrt2.o and the crt objects itself and hands ld absolute paths.
	PVCore::setenv("POCL_LINKER_DIR", libdir.c_str(), 1);
#endif

	size_t size = PVParallelView::MaxBciCodes * sizeof(PVBCICodeBase);
	int dev_idx = 0;
	cl_int err;
	const cl_uint Bbits = PARALLELVIEW_ZZT_BBITS;
	const cl_uint image_height = PVParallelView::constants<Bbits>::image_height;
	const size_t column_mem_size = image_height * sizeof(cl_uint);
	const uint64_t max_mem = column_mem_size * PARALLELVIEW_ZONE_MAX_WIDTH;

	const bool force_cpu = PVOpenCL::force_cpu();

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
 * PVParallelView::PVBCIDrawingBackendOpenCL::precompile_kernels
 *****************************************************************************/

void PVParallelView::PVBCIDrawingBackendOpenCL::precompile_kernels(
    const std::function<void(size_t, size_t)>& progress)
{
	if (_devices.empty()) {
		return;
	}

	// The work-group shape is rounded up to a power of two (see opencl_kernel),
	// so these are all the shapes the views can ask for: the widths a zone can
	// take are clamped to [ZoneMinWidth, ZoneMaxWidth]. Both image heights are
	// covered because the shape is capped by the local memory a column needs,
	// which the taller one exhausts sooner.
	std::vector<std::pair<size_t, uint8_t>> shapes;
	for (int height_bits : {PARALLELVIEW_ZT_BBITS, PARALLELVIEW_ZZT_BBITS}) {
		for (size_t width = PARALLELVIEW_ZONE_MIN_WIDTH; width <= PARALLELVIEW_ZONE_MAX_WIDTH;
		     width *= 2) {
			shapes.emplace_back(width, static_cast<uint8_t>(height_bits));
		}
	}

	size_t done = 0;

	for (const auto& [width, height_bits] : shapes) {
		if (progress) {
			progress(done, shapes.size());
		}

		// No codes to draw: the kernel is enqueued with the dimensions that
		// select the specialisation, and returns having written a blank image.
		PVBCIBackendImage_p image = create_image(width, height_bits);
		render(image, 0, width, nullptr, 0);
		wait_all();

		++done;
	}

	if (progress) {
		progress(done, shapes.size());
	}
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
		// sizeof(*codes), not sizeof(codes): the latter is the size of the
		// pointer, which only happens to match on the platforms built for.
		err = dev.queue.enqueueWriteBuffer(dev.buffer, CL_FALSE, 0, n * sizeof(*codes), codes);
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
