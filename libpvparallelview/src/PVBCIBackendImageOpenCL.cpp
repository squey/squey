/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

#include <pvkernel/opencl/common.h>

#include <pvparallelview/PVBCIBackendImageOpenCL.h>

#include <cassert>
#include <iostream>

/******************************************************************************
 * PVParallelView::PVBCIBackendImageOpenCL::PVBCIBackendImageOpenCL
 *****************************************************************************/

PVParallelView::PVBCIBackendImageOpenCL::PVBCIBackendImageOpenCL(const uint32_t width,
                                                                 uint8_t height_bits,
                                                                 const cl_context context,
                                                                 const cl_command_queue queue,
                                                                 int index)
    : PVBCIBackendImage(width, height_bits), _width(width), _queue(queue), _index(index)
{
	size_t size = PVBCIBackendImage::size_pixel() * sizeof(pixel_t);
	cl_int err;

	_host_addr = PVOpenCL::host_allocate<pixel_t>(context, _queue, CL_MEM_WRITE_ONLY, CL_MAP_READ,
	                                              size, _host_mem, err);
	inendi_verify_opencl_var(err);

	_device_mem = PVOpenCL::allocate(context, CL_MEM_READ_ONLY, size, err);
	inendi_verify_opencl_var(err);
}

/******************************************************************************
 * PVParallelView::PVBCIBackendImageOpenCL::~PVBCIBackendImageOpenCL
 *****************************************************************************/

PVParallelView::PVBCIBackendImageOpenCL::~PVBCIBackendImageOpenCL()
{
	inendi_verify_opencl(PVOpenCL::host_free(_queue, _host_mem, _host_addr));
	inendi_verify_opencl(PVOpenCL::free(_device_mem));
}

/******************************************************************************
 * PVParallelView::PVBCIBackendImageOpenCL::opy_device_to_host_async
 *****************************************************************************/

void PVParallelView::PVBCIBackendImageOpenCL::copy_device_to_host_async(cl_event* event) const
{
	inendi_verify_opencl(clEnqueueReadBuffer(_queue, _device_mem, CL_FALSE, 0UL,
	                                         size_pixel() * sizeof(pixel_t), _host_addr, 0U,
	                                         nullptr, event));
}

/******************************************************************************
 * PVParallelView::PVBCIBackendImageOpenCL::copy_device_to_host_sync
 *****************************************************************************/

void PVParallelView::PVBCIBackendImageOpenCL::copy_device_to_host_sync() const
{
	inendi_verify_opencl(clEnqueueReadBuffer(_queue, _device_mem, CL_TRUE, 0UL,
	                                         size_pixel() * sizeof(pixel_t), _host_addr, 0U,
	                                         nullptr, nullptr));
}

/******************************************************************************
 * PVParallelView::PVBCIBackendImageOpenCL::qimage
 *****************************************************************************/

QImage PVParallelView::PVBCIBackendImageOpenCL::qimage(size_t crop_height) const
{
	assert(crop_height <= PVBCIBackendImage::height());

	return QImage((const uchar*)_host_addr, PVBCIBackendImage::width(), crop_height,
	              _width * sizeof(uint32_t), QImage::Format_ARGB32_Premultiplied);
}
