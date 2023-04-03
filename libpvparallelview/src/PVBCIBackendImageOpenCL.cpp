//
// MIT License
//
// © ESI Group, 2015
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

	return {(const uchar*)_host_addr, static_cast<int32_t>(PVBCIBackendImage::width()), static_cast<int>(crop_height),
	              static_cast<int>(_width * sizeof(uint32_t)), QImage::Format_ARGB32_Premultiplied};

}
