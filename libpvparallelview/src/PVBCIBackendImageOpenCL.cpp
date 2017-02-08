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
                                                                 const uint8_t height_bits,
                                                                 const cl::Context& context,
                                                                 const cl::CommandQueue& queue,
                                                                 int index)
    : PVBCIBackendImage(width, height_bits), _width(width), _queue(queue), _index(index)
{
	size_t size = PVBCIBackendImage::size_pixel() * sizeof(pixel_t);
	cl_int err;

	_host_addr = PVOpenCL::host_allocate<pixel_t>(context, _queue, CL_MEM_WRITE_ONLY, CL_MAP_READ,
	                                              size, _host_buffer, err);
	inendi_verify_opencl_var(err);

	_device_buffer = cl::Buffer(context, CL_MEM_READ_ONLY, size, nullptr, &err);
	inendi_verify_opencl_var(err);
}

/******************************************************************************
 * PVParallelView::PVBCIBackendImageOpenCL::~PVBCIBackendImageOpenCL
 *****************************************************************************/

PVParallelView::PVBCIBackendImageOpenCL::~PVBCIBackendImageOpenCL()
{
	// no "cl::MappedBuffer" class, 'have to unmap it manually...
	inendi_verify_opencl(_queue.enqueueUnmapMemObject(_host_buffer, _host_addr));
}

/******************************************************************************
 * PVParallelView::PVBCIBackendImageOpenCL::opy_device_to_host_async
 *****************************************************************************/

void PVParallelView::PVBCIBackendImageOpenCL::copy_device_to_host_async(cl::Event* event) const
{
	inendi_verify_opencl(_queue.enqueueReadBuffer(
	    _device_buffer, CL_FALSE, 0, size_pixel() * sizeof(pixel_t), _host_addr, nullptr, event));
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
