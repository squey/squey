/**
 * \file bccb.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <common/common.h>
#include <common/bench.h>

#include <ocl/bccb.h>
#include <ocl/utils.h>

#include <CL/cl.h>

#define verify(e) __verify(e, __FILE__, __LINE__)
#define __verify(e, F, L)\
	if (!(e)) {\
		fprintf(stderr, "valid assertion failed at %s:%d: %s.\n", F, L, #e);\
		abort();\
	}

#define LOCAL_WORK_SIZE 1

void ocl_bccb(const char* kernels_file, PVBCode* bcodes, size_t n, BCodeCB bccb)
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

	// allocate the buffer memory objects
	cl_mem mem_bcodes, mem_bccb;
	mem_bcodes = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(PVBCode)*n, bcodes, NULL);
	mem_bccb = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, SIZE_BCODECB, bccb, NULL);
	verify(mem_bcodes != 0);
	verify(mem_bccb != 0);

	// create the program
	char* program_source = ReadSources(kernels_file);
	//size_t kern_size;
	//char* prog_binary = ReadBinary(kernels_file, kern_size);
	program = clCreateProgramWithSource(context, 1, (const char**)&program_source, NULL, NULL);
	//program = clCreateProgramWithBinary(context, 1, devices, &kern_size, (const unsigned char**)&prog_binary, NULL, NULL);
	verify(program);
	//free(prog_binary);
	free(program_source);

	// build the program
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	verify(err == CL_SUCCESS);

	// create the kernel
	kernel = clCreateKernel(program, "bccb_naive", NULL);
	verify(kernel != 0);

	{
		cl_ulong size;
		clGetDeviceInfo(devices[0], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &size, 0);
		printf("size max local mem: %lu\n", size);
	}

	// set the args values
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &mem_bcodes);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &mem_bccb);
	//err |= clSetKernelArg(kernel, 2, 10, NULL);

	verify(err == CL_SUCCESS);

	// set work-item dimensions
	global_work_size[0] = n/LOCAL_WORK_SIZE;
	local_work_size[0]= LOCAL_WORK_SIZE;

	clFinish(cmd_queue);

	// execute kernel
	BENCH_START(ocl);
	err = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	clFinish(cmd_queue);
	BENCH_END(ocl, "ocl-naive", n, sizeof(PVBCode), 1, SIZE_BCODECB);
	verify(err == CL_SUCCESS);

	// read output image
	err = clEnqueueReadBuffer(cmd_queue, mem_bccb, CL_TRUE, 0, SIZE_BCODECB, bccb, 0, NULL, NULL);
	verify(err == CL_SUCCESS);

	//release kernel, program, and memory objects
	clReleaseMemObject(mem_bcodes);
	clReleaseMemObject(mem_bccb);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmd_queue);
	clReleaseContext(context);
}
