//
// MIT License
//
// © Squey, 2026
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

/**
 * An OpenCL driver for the tests of the OpenCL start-up, loaded by the ICD
 * loader like any other.
 *
 * It offers a single GPU, and behaves as the Adreno driver of Windows on ARM
 * machines does: a device type made of several bits, such as
 * CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, is refused with
 * CL_INVALID_DEVICE_TYPE (BugSplat crashes 231 and 234).
 *
 * It also hands out a context on that GPU but no command queue, which makes it
 * a driver found usable that fails right after: the device can never be
 * brought up. The GPU says it is a dedicated one, and those are tried before
 * any GPU built into a processor: Topencl_gpu_fallback and
 * Tpvguiqt_about_box_opencl count on it being found first, whatever the
 * machine running them has.
 *
 * Built with SQUEY_TEST_ICD_HYBRID, it stands for the two GPUs of a hybrid
 * laptop instead: two platforms of a GPU each, the first built into the
 * processor, the second dedicated.
 *
 * Only what the loader and the start-up reach is answered; the other entries
 * of the dispatch table stay null.
 */

#include <CL/cl_icd.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <string_view>

struct _cl_platform_id {
	const cl_icd_dispatch* dispatch;
};

struct _cl_device_id {
	const cl_icd_dispatch* dispatch;
};

struct _cl_context {
	const cl_icd_dispatch* dispatch;
	cl_uint reference_count;
	cl_device_id device;
};

namespace
{

using notify_t = void(CL_CALLBACK*)(const char*, const void*, size_t, void*);

extern const cl_icd_dispatch dispatch;

struct gpu_spec_t {
	std::string_view platform_name;
	std::string_view device_name;
	cl_bool host_unified_memory;
};

// The GPUs on offer, a platform each, in the order the driver lists them
#ifdef SQUEY_TEST_ICD_HYBRID
constexpr gpu_spec_t gpu_specs[] = {
    {"Squey test integrated platform", "Squey test integrated GPU", CL_TRUE},
    {"Squey test dedicated platform", "Squey test dedicated GPU", CL_FALSE}};
_cl_platform_id platforms[] = {{&dispatch}, {&dispatch}};
_cl_device_id gpus[] = {{&dispatch}, {&dispatch}};
#else
constexpr gpu_spec_t gpu_specs[] = {{"Squey test platform", "Squey test GPU", CL_FALSE}};
_cl_platform_id platforms[] = {{&dispatch}};
_cl_device_id gpus[] = {{&dispatch}};
#endif

constexpr cl_uint gpu_count = std::size(gpu_specs);

// The position of a platform of this driver, gpu_count for any other
cl_uint index_of(cl_platform_id id)
{
	for (cl_uint i = 0; i < gpu_count; ++i) {
		if (id == &platforms[i]) {
			return i;
		}
	}
	return gpu_count;
}

// The position of a device of this driver, gpu_count for any other
cl_uint index_of(cl_device_id id)
{
	for (cl_uint i = 0; i < gpu_count; ++i) {
		if (id == &gpus[i]) {
			return i;
		}
	}
	return gpu_count;
}

cl_int answer(size_t size, void* value, size_t* size_ret, const void* data, size_t data_size)
{
	if (value != nullptr) {
		if (size < data_size) {
			return CL_INVALID_VALUE;
		}
		std::memcpy(value, data, data_size);
	}
	if (size_ret != nullptr) {
		*size_ret = data_size;
	}
	return CL_SUCCESS;
}

cl_int answer(size_t size, void* value, size_t* size_ret, std::string_view string)
{
	// The terminating null is part of the answer, and every string_view below
	// is built from a literal, which has one.
	return answer(size, value, size_ret, string.data(), string.size() + 1);
}

template <typename T>
cl_int answer_value(size_t size, void* value, size_t* size_ret, const T& data)
{
	return answer(size, value, size_ret, &data, sizeof(data));
}

bool is_single_type(cl_device_type type)
{
	switch (type) {
	case CL_DEVICE_TYPE_DEFAULT:
	case CL_DEVICE_TYPE_CPU:
	case CL_DEVICE_TYPE_GPU:
	case CL_DEVICE_TYPE_ACCELERATOR:
	case CL_DEVICE_TYPE_CUSTOM:
	case CL_DEVICE_TYPE_ALL:
		return true;
	default:
		return false;
	}
}

bool matches_gpu(cl_device_type type)
{
	return type == CL_DEVICE_TYPE_DEFAULT or (type & CL_DEVICE_TYPE_GPU) != 0;
}

cl_int CL_API_CALL get_platform_ids(cl_uint num_entries, cl_platform_id* ids, cl_uint* num_platforms)
{
	if (ids == nullptr and num_platforms == nullptr) {
		return CL_INVALID_VALUE;
	}
	if (ids != nullptr) {
		if (num_entries == 0) {
			return CL_INVALID_VALUE;
		}
		for (cl_uint i = 0; i < std::min(num_entries, gpu_count); ++i) {
			ids[i] = &platforms[i];
		}
	}
	if (num_platforms != nullptr) {
		*num_platforms = gpu_count;
	}
	return CL_SUCCESS;
}

cl_int CL_API_CALL get_platform_info(cl_platform_id id,
                                     cl_platform_info name,
                                     size_t size,
                                     void* value,
                                     size_t* size_ret)
{
	const cl_uint index = index_of(id);
	if (index == gpu_count) {
		return CL_INVALID_PLATFORM;
	}

	switch (name) {
	case CL_PLATFORM_PROFILE:
		return answer(size, value, size_ret, "FULL_PROFILE");
	case CL_PLATFORM_VERSION:
		// cl.hpp parses the version out of this to decide whether devices are
		// reference counted
		return answer(size, value, size_ret, "OpenCL 3.0 Squey test ICD");
	case CL_PLATFORM_NAME:
		return answer(size, value, size_ret, gpu_specs[index].platform_name);
	case CL_PLATFORM_VENDOR:
		return answer(size, value, size_ret, "Squey");
	case CL_PLATFORM_EXTENSIONS:
		return answer(size, value, size_ret, "cl_khr_icd");
	case CL_PLATFORM_ICD_SUFFIX_KHR:
		return answer(size, value, size_ret, "SQUEYTEST");
	default:
		return CL_INVALID_VALUE;
	}
}

cl_int CL_API_CALL get_device_ids(cl_platform_id id,
                                  cl_device_type type,
                                  cl_uint num_entries,
                                  cl_device_id* devices,
                                  cl_uint* num_devices)
{
	const cl_uint index = index_of(id);
	if (index == gpu_count) {
		return CL_INVALID_PLATFORM;
	}
	if (not is_single_type(type)) {
		return CL_INVALID_DEVICE_TYPE;
	}
	if (devices == nullptr and num_devices == nullptr) {
		return CL_INVALID_VALUE;
	}
	if (not matches_gpu(type)) {
		return CL_DEVICE_NOT_FOUND;
	}
	if (devices != nullptr) {
		if (num_entries == 0) {
			return CL_INVALID_VALUE;
		}
		devices[0] = &gpus[index];
	}
	if (num_devices != nullptr) {
		*num_devices = 1;
	}
	return CL_SUCCESS;
}

cl_int CL_API_CALL
get_device_info(cl_device_id id, cl_device_info name, size_t size, void* value, size_t* size_ret)
{
	const cl_uint index = index_of(id);
	if (index == gpu_count) {
		return CL_INVALID_DEVICE;
	}

	switch (name) {
	case CL_DEVICE_TYPE:
		return answer_value(size, value, size_ret, cl_device_type{CL_DEVICE_TYPE_GPU});
	case CL_DEVICE_PLATFORM:
		return answer_value(size, value, size_ret, cl_platform_id{&platforms[index]});
	case CL_DEVICE_HOST_UNIFIED_MEMORY:
		return answer_value(size, value, size_ret, gpu_specs[index].host_unified_memory);
	case CL_DEVICE_NAME:
		return answer(size, value, size_ret, gpu_specs[index].device_name);
	case CL_DEVICE_VENDOR:
		return answer(size, value, size_ret, "Squey");
	case CL_DEVICE_VERSION:
		return answer(size, value, size_ret, "OpenCL 3.0");
	case CL_DRIVER_VERSION:
		return answer(size, value, size_ret, "1.0");
	case CL_DEVICE_PROFILE:
		return answer(size, value, size_ret, "FULL_PROFILE");
	case CL_DEVICE_LOCAL_MEM_SIZE:
		return answer_value(size, value, size_ret, cl_ulong{32768});
	default:
		return CL_INVALID_VALUE;
	}
}

cl_int CL_API_CALL retain_device(cl_device_id id)
{
	return index_of(id) != gpu_count ? CL_SUCCESS : CL_INVALID_DEVICE;
}

cl_int CL_API_CALL release_device(cl_device_id id)
{
	return index_of(id) != gpu_count ? CL_SUCCESS : CL_INVALID_DEVICE;
}

cl_context new_context(cl_device_id device, cl_int* err)
{
	if (err != nullptr) {
		*err = CL_SUCCESS;
	}
	return new _cl_context{&dispatch, 1, device};
}

cl_context fail_context(cl_int* err, cl_int code)
{
	if (err != nullptr) {
		*err = code;
	}
	return nullptr;
}

cl_context CL_API_CALL create_context(const cl_context_properties* /*properties*/,
                                      cl_uint num_devices,
                                      const cl_device_id* devices,
                                      notify_t /*notify*/,
                                      void* /*user_data*/,
                                      cl_int* err)
{
	if (num_devices == 0 or devices == nullptr) {
		return fail_context(err, CL_INVALID_VALUE);
	}
	for (cl_uint i = 0; i < num_devices; ++i) {
		if (index_of(devices[i]) == gpu_count) {
			return fail_context(err, CL_INVALID_DEVICE);
		}
	}
	return new_context(devices[0], err);
}

cl_context CL_API_CALL create_context_from_type(const cl_context_properties* properties,
                                                cl_device_type type,
                                                notify_t /*notify*/,
                                                void* /*user_data*/,
                                                cl_int* err)
{
	if (not is_single_type(type)) {
		return fail_context(err, CL_INVALID_DEVICE_TYPE);
	}
	if (not matches_gpu(type)) {
		return fail_context(err, CL_DEVICE_NOT_FOUND);
	}
	// The GPU of the platform the properties name, of the first one otherwise
	cl_uint index = 0;
	for (auto* property = properties; property != nullptr and *property != 0; property += 2) {
		if (property[0] == CL_CONTEXT_PLATFORM) {
			index = index_of(reinterpret_cast<cl_platform_id>(property[1]));
		}
	}
	if (index == gpu_count) {
		return fail_context(err, CL_INVALID_PLATFORM);
	}
	return new_context(&gpus[index], err);
}

cl_int CL_API_CALL retain_context(cl_context context)
{
	if (context == nullptr) {
		return CL_INVALID_CONTEXT;
	}
	++context->reference_count;
	return CL_SUCCESS;
}

cl_int CL_API_CALL release_context(cl_context context)
{
	if (context == nullptr) {
		return CL_INVALID_CONTEXT;
	}
	if (--context->reference_count == 0) {
		delete context;
	}
	return CL_SUCCESS;
}

cl_int CL_API_CALL get_context_info(
    cl_context context, cl_context_info name, size_t size, void* value, size_t* size_ret)
{
	if (context == nullptr) {
		return CL_INVALID_CONTEXT;
	}

	switch (name) {
	case CL_CONTEXT_REFERENCE_COUNT:
		return answer_value(size, value, size_ret, context->reference_count);
	case CL_CONTEXT_NUM_DEVICES:
		return answer_value(size, value, size_ret, cl_uint{1});
	case CL_CONTEXT_DEVICES:
		return answer_value(size, value, size_ret, context->device);
	case CL_CONTEXT_PROPERTIES:
		return answer(size, value, size_ret, nullptr, 0);
	default:
		return CL_INVALID_VALUE;
	}
}

cl_command_queue CL_API_CALL create_command_queue(cl_context /*context*/,
                                                  cl_device_id /*device*/,
                                                  cl_command_queue_properties /*properties*/,
                                                  cl_int* err)
{
	if (err != nullptr) {
		*err = CL_OUT_OF_RESOURCES;
	}
	return nullptr;
}

cl_command_queue CL_API_CALL create_command_queue_with_properties(
    cl_context /*context*/,
    cl_device_id /*device*/,
    const cl_queue_properties* /*properties*/,
    cl_int* err)
{
	if (err != nullptr) {
		*err = CL_OUT_OF_RESOURCES;
	}
	return nullptr;
}

void* CL_API_CALL get_extension_function_address(const char* name);

cl_icd_dispatch make_dispatch()
{
	cl_icd_dispatch table{};
	table.clGetPlatformIDs = get_platform_ids;
	table.clGetPlatformInfo = get_platform_info;
	table.clGetDeviceIDs = get_device_ids;
	table.clGetDeviceInfo = get_device_info;
	table.clCreateContext = create_context;
	table.clCreateContextFromType = create_context_from_type;
	table.clRetainContext = retain_context;
	table.clReleaseContext = release_context;
	table.clGetContextInfo = get_context_info;
	table.clCreateCommandQueue = create_command_queue;
	table.clGetExtensionFunctionAddress = get_extension_function_address;
	table.clRetainDevice = retain_device;
	table.clReleaseDevice = release_device;
	table.clCreateCommandQueueWithProperties = create_command_queue_with_properties;
	return table;
}

const cl_icd_dispatch dispatch = make_dispatch();

} // namespace

extern "C" {

__attribute__((visibility("default"))) CL_API_ENTRY cl_int CL_API_CALL
clIcdGetPlatformIDsKHR(cl_uint num_entries, cl_platform_id* platforms, cl_uint* num_platforms)
{
	return get_platform_ids(num_entries, platforms, num_platforms);
}

__attribute__((visibility("default"))) CL_API_ENTRY cl_int CL_API_CALL
clGetPlatformInfo(cl_platform_id id,
                  cl_platform_info name,
                  size_t size,
                  void* value,
                  size_t* size_ret)
{
	return get_platform_info(id, name, size, value, size_ret);
}

__attribute__((visibility("default"))) CL_API_ENTRY void* CL_API_CALL
clGetExtensionFunctionAddress(const char* name)
{
	return get_extension_function_address(name);
}

} // extern "C"

namespace
{

void* CL_API_CALL get_extension_function_address(const char* name)
{
	const std::string_view function(name);

	// The functions of this file rather than the exported ones: the address of
	// an exported function is resolved through the global scope first, where
	// the ICD loader defines a clGetPlatformInfo of its own.
	if (function == "clIcdGetPlatformIDsKHR") {
		return reinterpret_cast<void*>(&get_platform_ids);
	}
	if (function == "clGetPlatformInfo") {
		return reinterpret_cast<void*>(&get_platform_info);
	}
	return nullptr;
}

} // namespace
