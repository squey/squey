/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

#ifndef PVKERNEL_OPENCL_COMMON_H
#define PVKERNEL_OPENCL_COMMON_H

#include <CL/cl.h>

#include <vector>
#include <functional>
#include <stdexcept>

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

}

using device_func = std::function<void(cl_context, cl_device_id)>;

/**
 * find the first OpenCL plateform matching @p type and call @p f on each of its devices
 *
 * @param type the wanted OpenCL plateform type
 * @param f a function to call on each device of the found context
 *
 * @return a valid OpenCL context in case of success; nullptr othewise.
 */
cl_context find_first_usable_context(cl_device_type type, device_func const& f);

/**
 *
 * @param ctx the OpenCL context to allocate on
 * @param flags a allocation/usage bit-field
 * @param size the size to allocate in bytes
 * @param err a variable to save the err state
 *
 * @return the memory object ident
 */
cl_mem allocate(const cl_context ctx, const cl_mem_flags flags, const size_t size, cl_int& err);

/**
 * allocate a host accessible OpenCL buffer object
 *
 * @param ctx the OpenCL context to allocate on
 * @param queue the OpenCL command queue to use
 * @param mem_flags a allocation/usage bit-field for the memory object
 * @param map_flags a allocation/usage bit-field for the mapped memory block
 * @param size the size to allocate in bytes
 * @param the resulting OpenCL memory object
 * @param err a variable to save the err state
 *
 * @return the host memory mapped address
 */
void* host_alloc(const cl_context ctx,
                 const cl_command_queue queue,
                 const cl_mem_flags mem_flags,
                 const cl_map_flags map_flags,
                 const size_t size,
                 cl_mem& mem,
                 cl_int& err);

/**
 * allocate a typed memory on an OpenCL context
 *
 * @see PVOpenCL::host_alloc
 */
template <typename T>
T* host_allocate(const cl_context ctx,
                 cl_command_queue queue,
                 const cl_mem_flags mem_flags,
                 cl_map_flags map_flags,
                 size_t size,
                 cl_mem& mem,
                 cl_int& err)
{
	return static_cast<T*>(host_alloc(ctx, queue, mem_flags, map_flags, size, mem, err));
}

/**
 * free memory block
 *
 * @param mem the OpenCL memory object to free
 *
 * @return the operation resulting error code
 */
cl_int free(const cl_mem mem);

/**
 * free a host mapped memory block
 *
 * @param queue the OpenCL command queue to use
 * @param mem the OpenCL memory object to free
 * @param addr the host mapped memory address
 *
 * @return the operation resulting error code
 */
cl_int host_free(const cl_command_queue queue, const cl_mem mem, void* addr);
}

#endif // PVKERNEL_OPENCL_COMMON_H
