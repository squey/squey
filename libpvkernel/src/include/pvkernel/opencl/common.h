/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

#ifndef PVKERNEL_OPENCL_COMMON_H
#define PVKERNEL_OPENCL_COMMON_H

#include <CL/cl.hpp> // for Buffer (ptr only), etc

#include <cstddef>    // for size_t
#include <cstdio>     // for fprintf, stderr
#include <cstdlib>    // for abort
#include <functional> // for function
#include <stdexcept>  // for runtime_error

#define __inendi_verify_opencl(E, F, L)                                                            \
	{                                                                                              \
		cl_int e = E;                                                                              \
		if ((e) != CL_SUCCESS) {                                                                   \
			fprintf(stderr, "OpenCL assert failed at %s:%d with error code %d\n", F, L, e);        \
			abort();                                                                               \
		}                                                                                          \
	}

#define __inendi_verify_opencl_var(E, F, L)                                                        \
	if ((E) != CL_SUCCESS) {                                                                       \
		fprintf(stderr, "OpenCL assert failed at %s:%d with error code %d\n", F, L, E);            \
		abort();                                                                                   \
	}

#define inendi_verify_opencl(E) __inendi_verify_opencl(E, __FILE__, __LINE__)
#define inendi_verify_opencl_var(E) __inendi_verify_opencl_var(E, __FILE__, __LINE__)

namespace PVOpenCL
{

namespace exception
{

class no_backend_error : public std::runtime_error
{
  public:
	no_backend_error() : std::runtime_error::runtime_error("No OpenCL backend found") {}
};
} // namespace exception

/**
 * the function type used as find_first_usable_context(...) parameter
 */
using device_func = std::function<void(cl::Context&, cl::Device&)>;

/**
 * @return a string containing version and devices used
 */
std::string opencl_version();

/**
 * find the first OpenCL plateform matching @p accelerated and call @p f on each of its devices
 *
 * @param accelerated a boolean to indicate if the found backend must use decidated hardware or must
 * used software implementation
 * @param f a function to call on each device of the found context
 *
 * @return a valid OpenCL context in case of success; a null context otherwise.
 */
cl::Context find_first_usable_context(bool accelerated, device_func const& f);

/**
 * allocate a host accessible OpenCL buffer object
 *
 * @param ctx the OpenCL context to allocate on
 * @param queue the OpenCL command queue to use
 * @param mem_flags a allocation/usage bit-field for the memory object
 * @param map_flags a allocation/usage bit-field for the mapped memory block
 * @param size the size to allocate in bytes
 * @param buffer the resulting OpenCL memory buffer
 * @param err a variable to save the err state
 *
 * @return the host memory mapped address
 */
void* host_alloc(const cl::Context& ctx,
                 const cl::CommandQueue& queue,
                 const cl_mem_flags mem_flags,
                 const cl_map_flags map_flags,
                 const size_t size,
                 cl::Buffer& buffer,
                 cl_int& err);

/**
 * allocate a typed memory on an OpenCL context
 *
 * @see PVOpenCL::host_alloc
 */
template <typename T>
T* host_allocate(const cl::Context& ctx,
                 const cl::CommandQueue& queue,
                 const cl_mem_flags mem_flags,
                 const cl_map_flags map_flags,
                 size_t size,
                 cl::Buffer& buffer,
                 cl_int& err)
{
	return static_cast<T*>(host_alloc(ctx, queue, mem_flags, map_flags, size, buffer, err));
}
} // namespace PVOpenCL

#endif // PVKERNEL_OPENCL_COMMON_H
