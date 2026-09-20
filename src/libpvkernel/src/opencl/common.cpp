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
#include <utility> // for pair
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

	/* Every platform with devices of the type sought, and those devices, in the
	 * order the loader lists the platforms.
	 */
	std::vector<std::pair<cl::Platform, std::vector<cl::Device>>> candidates;

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

		candidates.emplace_back(platform, std::move(devices));
	}

	/* A hybrid laptop offers two GPUs, usually on two platforms: the one built
	 * into its processor and a dedicated one, several times faster at drawing the
	 * views -- 2.8 times, an RTX 3060 against an Iris Xe. Nothing lists the
	 * platforms by speed: ocl-icd puts first those with the most GPUs, and leaves
	 * the others in the order its directory reads. So the dedicated GPUs get a
	 * round of their own first. A platform picked by index in the settings is
	 * taken as it comes.
	 */
	const bool dedicated_first = accelerated and wanted_platform_index == PLATFORM_ANY_INDEX;

	for (const bool dedicated_only : {true, false}) {
		if (dedicated_only and not dedicated_first) {
			continue;
		}

		int platform_index = 0;

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
