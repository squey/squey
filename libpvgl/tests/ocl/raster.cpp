#include <common/common.h>
#include <common/bench.h>

#include <ocl/raster.h>
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

void ocl_raster(const char* kernels_file, const float* yl, const float* yr, const size_t n, unsigned int* img_idxes, const float zoom_x, const float zoom_y)
{
	cl_context context;
	cl_command_queue cmd_queue;
	cl_program program;
	cl_kernel kernel;
	size_t global_work_size[2];
	size_t local_work_size[2];
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
	kernel = clCreateKernel(program, "raster_nolocal", NULL);
	verify(kernel != 0);

	// map static memory objects
	memset(img_idxes, 0xFF, SIZE_GLOBAL_IDX_TABLE);
	cl_mem mem_img_idxes = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, SIZE_GLOBAL_IDX_TABLE, img_idxes, NULL);
	cl_mem mem_yl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float)*n, (void*) yl, NULL);
	cl_mem mem_yr = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float)*n, (void*) yr, NULL);
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &mem_yl);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &mem_yr);
	verify(err == CL_SUCCESS);

	// set the static args values
	err = clSetKernelArg(kernel, 2, sizeof(unsigned int), (void *) &n);
	err |= clSetKernelArg(kernel, 4, sizeof(float), (void *) &zoom_x);
	err |= clSetKernelArg(kernel, 5, sizeof(float), (void *) &zoom_y);
	err |= clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *) &mem_img_idxes);
	//err |= clSetKernelArg(kernel, 7, SIZE_LOCAL_IDX_TABLE, NULL);

	verify(err == CL_SUCCESS);

	// set work-item dimensions
	global_work_size[0] = IMAGE_WIDTH;
	global_work_size[1] = IMAGE_HEIGHT;
	local_work_size[0]= LOCAL_IDX_WIDTH;
	local_work_size[1]= LOCAL_IDX_HEIGHT;

	printf("Nb passes: %d\n", n/NLINES_PER_KERNEL);
	const unsigned int lend = (n/NLINES_PER_KERNEL)*NLINES_PER_KERNEL;
	clFinish(cmd_queue);
	unsigned int offset = NLINES_PER_KERNEL;
	err = clSetKernelArg(kernel, 3, sizeof(unsigned int), (void*) &offset);
	BENCH_START(ocl);
#if 0
	for (unsigned int offset = 0; offset < lend; offset += NLINES_PER_KERNEL)
	{
		err = clSetKernelArg(kernel, 3, sizeof(unsigned int), (void*) &offset);
		verify(err == CL_SUCCESS);

		err = clEnqueueNDRangeKernel(cmd_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
		verify(err == CL_SUCCESS);
	}
	if (lend < n) {
		err = clSetKernelArg(kernel, 3, sizeof(unsigned int), (void*) &lend);
		unsigned int lastb = n-lend;
		err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), (void*) &lastb);
		verify(err == CL_SUCCESS);

		err = clEnqueueNDRangeKernel(cmd_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
		verify(err == CL_SUCCESS);
	}
#endif
	clEnqueueNDRangeKernel(cmd_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	verify(err == CL_SUCCESS);
	clFinish(cmd_queue);
	//clFinish(cmd_queue);
	BENCH_END(ocl, "ocl raster", n, 2*sizeof(float), 1, SIZE_GLOBAL_IDX_TABLE);

#if 0
	memset(img_idxes, 0xFF, SIZE_GLOBAL_IDX_TABLE);
	clReleaseMemObject(mem_img_idxes);
	mem_img_idxes = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, SIZE_GLOBAL_IDX_TABLE, img_idxes, NULL);
	clFinish(cmd_queue);

	clReleaseKernel(kernel);
	kernel = clCreateKernel(program, "raster_nolocal", NULL);
	verify(kernel != 0);

	// read output image indexes
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &mem_yl);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &mem_yr);
	err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), (void *) &nlines_per_kernel);
	err |= clSetKernelArg(kernel, 4, sizeof(float), (void *) &zoom_x);
	err |= clSetKernelArg(kernel, 5, sizeof(float), (void *) &zoom_y);
	err |= clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *) &mem_img_idxes);
	verify(err == CL_SUCCESS);

	memset(img_idxes, 0xFF, SIZE_GLOBAL_IDX_TABLE);
	clFinish(cmd_queue);
	BENCH_START(ocl_nol);
	for (unsigned int offset = 0; offset < lend; offset += NLINES_PER_KERNEL)
	{
		err = clSetKernelArg(kernel, 3, sizeof(unsigned int), (void*) &offset);
		verify(err == CL_SUCCESS);

		err = clEnqueueNDRangeKernel(cmd_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
		verify(err == CL_SUCCESS);
	}
	clFinish(cmd_queue);
	BENCH_END(ocl_nol, "ocl raster nolocal", lend, 2*sizeof(float), 1, SIZE_GLOBAL_IDX_TABLE);

	verify(err == CL_SUCCESS);
#endif

	clEnqueueReadBuffer(cmd_queue, mem_img_idxes, CL_TRUE, 0, SIZE_GLOBAL_IDX_TABLE, (void*) img_idxes, 0, NULL, NULL);
	clFinish(cmd_queue);
	FILE* f = fopen("image.raw", "w+");
	fwrite(img_idxes, 1, SIZE_GLOBAL_IDX_TABLE, f);
	fclose(f);

	//release kernel, program, and memory objects
	clReleaseMemObject(mem_img_idxes);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmd_queue);
	clReleaseContext(context);
}
