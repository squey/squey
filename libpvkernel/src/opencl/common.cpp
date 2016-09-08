/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

#include <pvkernel/opencl/common.h> // for inendi_verify_opencl_var, etc

#include <pvkernel/core/PVConfig.h> // for PVConfig
#include <pvkernel/core/PVLogger.h> // for PVLOG_INFO

#include <CL/cl.hpp> // for cl_context_properties, etc

#include <cstddef> // for size_t
#include <string>  // for string
#include <vector>  // for vector

#include <QSettings>
#include <QVariant>

static const constexpr int PLATFORM_ANY_INDEX = -1;

/*****************************************************************************
 * PVOpenCL::visit_usable_devices
 *****************************************************************************/

cl::Context PVOpenCL::find_first_usable_context(bool accelerated, PVOpenCL::device_func const& f)
{
	cl_int err;

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
		return cl::Context();
	}

	auto& config = PVCore::PVConfig::get().config();
	const int wanted_platform_index =
	    config.value("backend_opencl/platform_index", PLATFORM_ANY_INDEX).toInt();

	int platform_index = 0;

	for (const auto& platform : platforms) {
		cl_context_properties prop[] = {CL_CONTEXT_PLATFORM,
		                                reinterpret_cast<cl_context_properties>(platform()), 0};

		cl::Context ctx(type, prop, nullptr, nullptr, &err);

		if (err == CL_DEVICE_NOT_FOUND) {
			// there is no matching device
			continue;
		}

		inendi_verify_opencl_var(err);

		if ((wanted_platform_index != PLATFORM_ANY_INDEX) &&
		    (platform_index != wanted_platform_index)) {
			++platform_index;
			continue;
		}

		std::vector<cl::Device> devices = ctx.getInfo<CL_CONTEXT_DEVICES>(&err);
		inendi_verify_opencl_var(err);

		if (devices.size() != 0) {
			for (auto& device : devices) {
				f(ctx, device);
			}

			std::string pname = platform.getInfo<CL_PLATFORM_NAME>(&err);
			inendi_verify_opencl_var(err);

			PVLOG_INFO("OpenCL backend found: %s\n", pname.c_str());

			return ctx;
		}

		++platform_index;
	}

	PVLOG_INFO("No %s OpenCL backend found\n", type_name);

	return cl::Context();
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
