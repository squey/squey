//
// MIT License
//
// © Squey, 2026
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
 * Checks that a GPU the backend fails to set up leaves the drawing to the CPU
 * device rather than to the QPainter backend.
 *
 * The driver loaded here hands out a context on its GPU but no command queue
 * (see opencl_test_icd.cpp in the tests of libpvkernel). ocl-icd lists the
 * drivers named by OCL_ICD_FILENAMES before those of the vendor files, and
 * OCL_ICD_PLATFORM_SORT=none keeps that order, so that this GPU is the one
 * found whatever else the machine has. PortableCL comes from the vendor files,
 * as it does for Topencl_cpu_device.
 *
 * FORCE_CPU is cleared rather than read from the environment: the GPU has to be
 * tried for its failure to matter.
 */
int main()
{
	PVCore::setenv("OCL_ICD_FILENAMES", SQUEY_TEST_OPENCL_ICD, 1);
	PVCore::setenv("OCL_ICD_PLATFORM_SORT", "none", 1);
	PVCore::setenv("FORCE_CPU", "0", 1);

	auto& backend = PVParallelView::PVBCIDrawingBackendOpenCL::get();

	// No device means PVParallelViewImpl would take the QPainter backend.
	PV_ASSERT_VALID(backend.device_count() > 0, "device count", backend.device_count());
	PV_VALID(backend.is_gpu_accelerated(), false);

	// What the backend reports drawing on is what it set up, the GPU left out.
	const auto devices = backend.opencl_devices();
	PV_VALID(devices.size(), backend.device_count());
	for (const auto& device : devices) {
		PV_ASSERT_VALID(device.name != "Squey test GPU", "device", device.name);
	}

	return 0;
}
