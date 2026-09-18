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

#include <pvkernel/core/PVConfig.h> // for PVConfig
#include <pvkernel/core/PVLogger.h> // for PVLOG_INFO

#include <CL/cl.hpp> // for cl_context_properties, etc

#include <cstddef> // for size_t
#include <cstdlib> // for getenv
#include <string>  // for string
#include <vector>  // for vector

#include <QSettings>
#include <QVariant>

#include <pvlogger.h>

static const constexpr int PLATFORM_ANY_INDEX = -1;

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

/* The platform and device strings below only feed the log. A driver that will
 * not describe itself earns a line in the log rather than the loss of the
 * device it was asked about.
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

} // namespace

bool PVOpenCL::force_cpu()
{
	if (PVCore::PVConfig::get().config().value("backend_opencl/force_cpu", false).toBool()) {
		return true;
	}
	const char* env = getenv("FORCE_CPU");
	return env != nullptr && std::string(env) == "1";
}

/*****************************************************************************
 * PVOpenCL::visit_usable_devices
 *****************************************************************************/

cl::Context PVOpenCL::find_first_usable_context(bool accelerated, PVOpenCL::device_func const& f)
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

	auto& config = PVCore::PVConfig::get().config();
	const int wanted_platform_index =
	    config.value("backend_opencl/platform_index", PLATFORM_ANY_INDEX).toInt();

	int platform_index = 0;

	for (const auto& platform : platforms) {
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

		cl_context_properties prop[] = {CL_CONTEXT_PLATFORM,
		                                reinterpret_cast<cl_context_properties>(platform()), 0};

		cl::Context ctx(devices, prop, nullptr, nullptr, &err);

		/* A platform that will not hand out a context is one to walk past, not
		 * a reason to end the process: the next platform, or the QPainter
		 * backend, can still draw the views.
		 */
		if (squey_opencl_failed(err)) {
			continue;
		}

		if ((wanted_platform_index != PLATFORM_ANY_INDEX) &&
		    (platform_index != wanted_platform_index)) {
			++platform_index;
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
