/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

#include <pvkernel/opencl/common.h>

#include <memory>
#include <iostream>

/*****************************************************************************
 * PVOpenCL::visit_usable_devices
 *****************************************************************************/

cl_context PVOpenCL::find_first_usable_context(cl_device_type type, PVOpenCL::device_func const& f)
{
	cl_context ctx;
	cl_int err;
	cl_uint pcount;
	cl_uint dcount;

	clGetPlatformIDs(0, nullptr, &pcount);
	if (pcount == 0) {
		return nullptr;
	}

	std::unique_ptr<cl_platform_id[]> ptab(new cl_platform_id[pcount]);
	std::unique_ptr<cl_device_id[]> dtab;

	clGetPlatformIDs(pcount, ptab.get(), nullptr);

	for (size_t i = 0; i < pcount; ++i) {
		size_t vsize;
		std::string pname;

		clGetPlatformInfo(ptab[i], CL_PLATFORM_NAME, 0, nullptr, &vsize);
		pname.resize(vsize);
		clGetPlatformInfo(ptab[i], CL_PLATFORM_NAME, vsize, &pname[0], nullptr);

		cl_context_properties prop[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)ptab[i], 0};

		ctx = clCreateContextFromType(prop, type, nullptr, nullptr, &err);

		if (err == CL_DEVICE_NOT_FOUND) {
			// there is no matching device
			continue;
		}

		inendi_verify_opencl_var(err);

		err = clGetContextInfo(ctx, CL_CONTEXT_NUM_DEVICES, sizeof(dcount), &dcount, NULL);
		inendi_verify_opencl_var(err);

		if (dcount != 0) {
			dtab.reset(new cl_device_id[dcount]);

			err = clGetContextInfo(ctx, CL_CONTEXT_DEVICES, sizeof(cl_device_id) * dcount,
			                       dtab.get(), nullptr);
			inendi_verify_opencl_var(err);

			for (size_t j = 0; j < dcount; ++j) {
				f(ctx, dtab[j]);
			}

			return ctx;
		}

		err = clReleaseContext(ctx);
		inendi_verify_opencl_var(err);
	}

	return nullptr;
}

/*****************************************************************************
 * PVOpenCL::allocate
 *****************************************************************************/

cl_mem
PVOpenCL::allocate(const cl_context ctx, const cl_mem_flags flags, const size_t size, cl_int& err)
{
	return clCreateBuffer(ctx, flags, size, nullptr, &err);
}

/*****************************************************************************
 * PVOpenCL::host_alloc
 *****************************************************************************/

void* PVOpenCL::host_alloc(const cl_context ctx,
                           const cl_command_queue queue,
                           const cl_mem_flags mem_flags,
                           const cl_map_flags map_flags,
                           const size_t size,
                           cl_mem& mem,
                           cl_int& err)
{
	mem = PVOpenCL::allocate(ctx, CL_MEM_ALLOC_HOST_PTR | mem_flags, size, err);

	if (err != CL_SUCCESS) {
		return nullptr;
	}

	return clEnqueueMapBuffer(queue, mem, CL_TRUE, map_flags, 0, size, 0, nullptr, nullptr, &err);
}

/*****************************************************************************
 * PVOpenCL::free
 *****************************************************************************/

cl_int PVOpenCL::free(const cl_mem mem)
{
	return clReleaseMemObject(mem);
}

/*****************************************************************************
 * PVOpenCL::host_free
 *****************************************************************************/

cl_int PVOpenCL::host_free(const cl_command_queue queue, const cl_mem mem, void* addr)
{
	cl_int err;

	err = clEnqueueUnmapMemObject(queue, mem, addr, 0, nullptr, nullptr);
	if (err != CL_SUCCESS) {
		return err;
	}

	return PVOpenCL::free(mem);
}
