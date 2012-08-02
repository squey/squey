/**
 * \file bccb2.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <common/common.h>
#include <common/bench.h>

#include <code_bz/bcode_cb.h>
#include <ocl/bccb.h>
#include <ocl/utils.h>

#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define verify(e) __verify(e, __FILE__, __LINE__)
#define __verify(e, F, L)\
	if (!(e)) {\
		fprintf(stderr, "valid assertion failed at %s:%d: %s.\n", F, L, #e);\
		abort();\
	}

#define NBCODES_PER_KERNEL (12*1024*1024/(2*sizeof(PVBCode)))

void ocl_bccb2(const char* kernels_file, const PVBCode* bcodes, const size_t n, BCodeCB bccb)
{
	cl_context context;
	cl_command_queue cmd_queue;
	cl_program program;
	cl_kernel kernel;
	size_t global_work_size[1];
	size_t local_work_size[1];
	size_t cb;
	cl_int err;

	cl_platform_id intel_platform_id = GetIntelOCLPlatform();
	verify(intel_platform_id != NULL);

	cl_context_properties context_properties[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)intel_platform_id, NULL };

	// create the OpenCL context on a CPU 
	context = clCreateContextFromType(context_properties, CL_DEVICE_TYPE_CPU, NULL, NULL, NULL);
	verify(context != 0);

	// get the list of CPU devices associated with context
	err = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &cb);
	cl_device_id *devices = (cl_device_id*)malloc(cb);
	verify(devices != NULL);
	clGetContextInfo(context, CL_CONTEXT_DEVICES, cb, devices, NULL);

	// create a command-queue
	cmd_queue = clCreateCommandQueue(context, devices[0], 0, NULL);
	verify(cmd_queue != 0);

	// create the program
	char* program_source = ReadSources(kernels_file);
	program = clCreateProgramWithSource(context, 1, (const char**)&program_source, NULL, NULL);
	verify(program);
	free(program_source);

	// build the program
	err = clBuildProgram(program, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);
	verify(err == CL_SUCCESS);

	// create the kernel
	kernel = clCreateKernel(program, "bccb2", NULL);
	verify(kernel != 0);

	// map static memory objects2
	memset(bccb, 0, SIZE_BCODECB);
	cl_mem mem_cb = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, SIZE_BCODECB, (void*) bccb, NULL);
	cl_mem mem_bcodes = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(PVBCode)*n, (void*) bcodes, NULL);
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &mem_bcodes);
	err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*) &mem_cb);
	verify(err == CL_SUCCESS);

	// set the static args values
	err = clSetKernelArg(kernel, 1, sizeof(unsigned int), (void *) &n);
	unsigned int offset = NBCODES_PER_KERNEL;
	err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), (void*) &offset);
	verify(err == CL_SUCCESS);

	// set work-item dimensions
	global_work_size[0] = NB_INT_BCODECB;
	local_work_size[0]= 16*16;

	clFinish(cmd_queue);
	BENCH_START(ocl);
	clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	verify(err == CL_SUCCESS);
	clFinish(cmd_queue);
	//clFinish(cmd_queue);
	BENCH_END(ocl, "ocl bccb2", n, sizeof(PVBCode), 1, SIZE_BCODECB);

	clEnqueueReadBuffer(cmd_queue, mem_bcodes, CL_TRUE, 0, SIZE_BCODECB, (void*) bccb, 0, NULL, NULL);
	clFinish(cmd_queue);

	//release kernel, program, and memory objects
	clReleaseMemObject(mem_cb);
	clReleaseMemObject(mem_bcodes);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmd_queue);
	clReleaseContext(context);
}
