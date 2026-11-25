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

#include <pvkernel/opencl/common.h> // for squey_verify_opencl_var, etc

#include <pvkernel/core/PVKernelSettings.h>
#include <pvkernel/core/PVLogger.h> // for PVLOG_INFO

#include <CL/cl.hpp> // for cl_context_properties, etc

#include <cstddef>  // for size_t
#include <cstdlib>  // for getenv
#include <iterator> // for make_move_iterator
#include <string>   // for string
#include <utility>  // for pair
#include <vector>   // for vector

#include <pvlogger.h>

bool PVOpenCL::failed(cl_int err, const char* file, int line)
{
	if (err == CL_SUCCESS) {
		return false;
	}

	PVLOG_WARN("OpenCL call failed at %s:%d with error code %d\n", file, line, err);

	return true;
}

namespace
{

/* The platform and device strings below feed the log, the list of the settings
 * dialog and the choice of a platform by its name. A driver that will not
 * describe itself earns a line in the log rather than the loss of the device it
 * was asked about, and cannot be chosen by its name.
 */
template <cl_int Param, typename Object>
auto info(const Object& object)
{
	cl_int err = CL_SUCCESS;
	auto value = object.template getInfo<Param>(&err);

	if (squey_opencl_failed(err)) {
		return decltype(value){};
	}

	return value;
}

/* A GPU built into the processor shares the memory of the host, a dedicated one
 * has its own: CL_DEVICE_HOST_UNIFIED_MEMORY tells them apart, and drivers still
 * answer it although OpenCL 2.0 deprecated it. A device whose driver does not
 * is not taken for a dedicated one.
 */
bool is_dedicated(const cl::Device& device)
{
	cl_int err = CL_SUCCESS;
	const cl_bool unified = device.getInfo<CL_DEVICE_HOST_UNIFIED_MEMORY>(&err);

	return err == CL_SUCCESS and unified == CL_FALSE;
}

} // namespace

PVOpenCL::lines_platform_t PVOpenCL::lines_platform()
{
	using kind_t = lines_platform_t::kind_t;

	const std::string value = PVCore::settings::lines_platform.value().toStdString();
	lines_platform_t platform{kind_t::automatic, {}};
	if (value == "qt") {
		platform.kind = kind_t::qt;
	} else if (value == "cpu") {
		platform.kind = kind_t::processor;
	} else if (value.starts_with("gpu:")) {
		platform = {kind_t::accelerated, value.substr(4)};
	} else if (value.starts_with("cpu:")) {
		platform = {kind_t::processor, value.substr(4)};
	} else if (value != "auto") {
		PVLOG_WARN("Unknown %s \"%s\", drawing as with \"auto\"\n",
		           qPrintable(PVCore::settings::lines_platform.key()), value.c_str());
	}

	const char* env = getenv("FORCE_CPU");
	const bool force_cpu = env != nullptr and std::string(env) == "1";
	if (force_cpu and
	    (platform.kind == kind_t::automatic or platform.kind == kind_t::accelerated)) {
		platform = {kind_t::processor, {}};
	}
	return platform;
}

std::vector<PVOpenCL::platform_t> PVOpenCL::platforms()
{
	std::vector<cl::Platform> cl_platforms;
	cl::Platform::get(&cl_platforms);

	std::vector<platform_t> accelerated;
	std::vector<platform_t> processors;
	for (const auto& platform : cl_platforms) {
		std::vector<cl::Device> devices;
		// All of them, sorted here: see find_first_usable_context() for the drivers that refuse
		// a combination of types.
		const cl_int err = platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
		if (err == CL_DEVICE_NOT_FOUND or squey_opencl_failed(err)) {
			continue;
		}

		const std::string name = info<CL_PLATFORM_NAME>(platform);
		platform_t gpus{name, true, {}};
		platform_t cpus{name, false, {}};
		for (const auto& device : devices) {
			const cl_device_type type = info<CL_DEVICE_TYPE>(device);
			if ((type & (CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR)) != 0) {
				gpus.devices.push_back(info<CL_DEVICE_NAME>(device));
			} else if ((type & CL_DEVICE_TYPE_CPU) != 0) {
				cpus.devices.push_back(info<CL_DEVICE_NAME>(device));
			}
		}
		if (not gpus.devices.empty()) {
			accelerated.push_back(std::move(gpus));
		}
		if (not cpus.devices.empty()) {
			processors.push_back(std::move(cpus));
		}
	}

	accelerated.insert(accelerated.end(), std::make_move_iterator(processors.begin()),
	                   std::make_move_iterator(processors.end()));
	return accelerated;
}

/*****************************************************************************
 * PVOpenCL::visit_usable_devices
 *****************************************************************************/

cl::Context PVOpenCL::find_first_usable_context(bool accelerated,
                                                PVOpenCL::device_func const& f,
                                                std::string const& platform_name)
{
	cl_int err = 0;

	cl_device_type type;
	const char* type_name;

	if (accelerated) {
		type = CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR;
		type_name = "accelerated";
	} else {
		type = CL_DEVICE_TYPE_CPU;
		type_name = "software";
	}

	std::vector<cl::Platform> platforms;

	cl::Platform::get(&platforms);

	if (platforms.size() == 0) {
		return {};
	}

	/* Every platform with devices of the type sought, and those devices, in the
	 * order the loader lists the platforms.
	 */
	std::vector<std::pair<cl::Platform, std::vector<cl::Device>>> candidates;

	for (const auto& platform : platforms) {
		// The drivers of the other platforms are not asked for anything more.
		if (not platform_name.empty() and info<CL_PLATFORM_NAME>(platform) != platform_name) {
			continue;
		}

		/* Ask the driver for all of its devices and match the type here, on what
		 * each device reports. The Adreno driver of Windows on ARM machines
		 * refuses CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR with
		 * CL_INVALID_DEVICE_TYPE, although the specification makes the type a
		 * bit-field: handed to clCreateContextFromType, it cost those machines
		 * their GPU, and before that the whole session. CL_DEVICE_TYPE_ALL is
		 * what every tool listing devices asks for, the one value a driver
		 * cannot get away with refusing.
		 */
		std::vector<cl::Device> platform_devices;
		err = platform.getDevices(CL_DEVICE_TYPE_ALL, &platform_devices);

		if (err == CL_DEVICE_NOT_FOUND) {
			// the platform has no device at all
			continue;
		}

		if (squey_opencl_failed(err)) {
			continue;
		}

		std::vector<cl::Device> devices;
		for (const auto& device : platform_devices) {
			if ((info<CL_DEVICE_TYPE>(device) & type) != 0) {
				devices.push_back(device);
			}
		}

		if (devices.empty()) {
			// there is no matching device
			continue;
		}

		candidates.emplace_back(platform, std::move(devices));
	}

	/* A hybrid laptop offers two GPUs, usually on two platforms: the one built
	 * into its processor and a dedicated one, several times faster at drawing the
	 * views -- 2.8 times, an RTX 3060 against an Iris Xe. Nothing lists the
	 * platforms by speed: ocl-icd puts first those with the most GPUs, and leaves
	 * the others in the order its directory reads. So the dedicated GPUs get a
	 * round of their own first. A platform picked by name in the settings is
	 * taken as it comes.
	 */
	const bool dedicated_first = accelerated and platform_name.empty();

	for (const bool dedicated_only : {true, false}) {
		if (dedicated_only and not dedicated_first) {
			continue;
		}

		for (const auto& [platform, platform_devices] : candidates) {
			std::vector<cl::Device> devices;
			for (const auto& device : platform_devices) {
				if (not dedicated_only or is_dedicated(device)) {
					devices.push_back(device);
				}
			}

			if (devices.empty()) {
				continue;
			}

			cl_context_properties prop[] = {
			    CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform()), 0};

			cl::Context ctx(devices, prop, nullptr, nullptr, &err);

			/* A platform that will not hand out a context is one to walk past, not
			 * a reason to end the process: the next platform, or the QPainter
			 * backend, can still draw the views.
			 */
			if (squey_opencl_failed(err)) {
				continue;
			}

			PVLOG_INFO("OpenCL backend found: %s, Version: %s, Vendor: %s, Profil: %s\n",
			           info<CL_PLATFORM_NAME>(platform).c_str(),
			           info<CL_PLATFORM_VERSION>(platform).c_str(),
			           info<CL_PLATFORM_VENDOR>(platform).c_str(),
			           info<CL_PLATFORM_PROFILE>(platform).c_str());

			PVLOG_INFO("OpenCL backend extensions: %s\n",
			           info<CL_PLATFORM_EXTENSIONS>(platform).c_str());

			for (auto& device : devices) {
				f(ctx, device);

				PVLOG_INFO("OpenCL device found: %s, Version: %s, Vendor: %s, Profil, %s LocalMemSize: %d\n",
				           info<CL_DEVICE_NAME>(device).c_str(),
				           info<CL_DEVICE_VERSION>(device).c_str(),
				           info<CL_DEVICE_VENDOR>(device).c_str(),
				           info<CL_DEVICE_PROFILE>(device).c_str(),
				           info<CL_DEVICE_LOCAL_MEM_SIZE>(device));
			}

			return ctx;
		}
	}

	PVLOG_INFO("No %s OpenCL backend found\n", type_name);

	return {};
}

/*****************************************************************************
 * PVOpenCL::host_alloc
 *****************************************************************************/

void* PVOpenCL::host_alloc(const cl::Context& ctx,
                           const cl::CommandQueue& queue,
                           const cl_mem_flags mem_flags,
                           const cl_map_flags map_flags,
                           const size_t size,
                           cl::Buffer& buffer,
                           cl_int& err)
{
	buffer = cl::Buffer(ctx, CL_MEM_ALLOC_HOST_PTR | mem_flags, size, nullptr, &err);

	if (err != CL_SUCCESS) {
		return nullptr;
	}

	return queue.enqueueMapBuffer(buffer, CL_TRUE, map_flags, 0, size, nullptr, nullptr, &err);
}
