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

#include <pvkernel/core/PVUtils.h>
#include <pvkernel/core/squey_assert.h>
#include <pvparallelview/PVBCIDrawingBackendOpenCL.h>

/**
 * Checks that OpenCL has a CPU device to offer.
 *
 * PVParallelViewImpl picks its drawing backend on device_count() and falls back
 * to the QPainter one -- the "degraded mode" the status bar warns about -- when
 * there is none. Nothing fails when that happens: the application starts, draws,
 * and is merely slower, so a packaging mistake that stops PortableCL from
 * registering itself as an ICD goes unnoticed. It did, on Linux, for as long as
 * the vendor file written at startup named a libpocl.so no build produced.
 *
 * Each platform reaches its CPU device its own way -- the vendor file on Linux,
 * OCL_ICD_FILENAMES set by the backend on Windows, PortableCL linked as the
 * OpenCL library itself on macOS -- which is reason enough to assert the result
 * everywhere rather than trust any of the three.
 *
 * FORCE_CPU is set here rather than read from the environment so that a runner
 * with a GPU asserts the same thing as one without.
 */
int main()
{
	PVCore::setenv("FORCE_CPU", "1", 1);

	auto& backend = PVParallelView::PVBCIDrawingBackendOpenCL::get();

	// No device means PVParallelViewImpl would take the QPainter backend.
	PV_ASSERT_VALID(backend.device_count() > 0, "device count", backend.device_count());
	PV_VALID(backend.is_gpu_accelerated(), false);

	return 0;
}
