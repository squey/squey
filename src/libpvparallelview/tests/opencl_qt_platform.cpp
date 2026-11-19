//
// MIT License
//
// © Squey, 2026
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/core/PVKernelSettings.h>
#include <pvkernel/core/squey_assert.h>
#include <pvparallelview/PVBCIDrawingBackendOpenCL.h>
#include <pvparallelview/PVParallelView.h>

/**
 * Checks that Qt, asked for by the settings, draws the lines of the parallel
 * coordinates although OpenCL has devices to offer: Topencl_cpu_device asserts
 * there is a CPU one wherever the tests run.
 */
int main()
{
	PV_ASSERT_VALID(PVCore::settings::lines_platform.set_value("qt"));

	PV_VALID(PVParallelView::PVBCIDrawingBackendOpenCL::get().device_count(), size_t{0});
	{
		PVParallelView::common::RAII_backend_init backend_resources;
		PV_ASSERT_VALID(PVParallelView::common::backend().opencl_devices().empty());
		PV_VALID(PVParallelView::common::is_gpu_accelerated(), false);
	}

	return 0;
}
