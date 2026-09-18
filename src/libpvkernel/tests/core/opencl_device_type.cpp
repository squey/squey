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
#include <pvkernel/opencl/common.h>

#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

/**
 * Checks that the OpenCL start-up gets its GPU out of a driver that refuses a
 * device type made of several bits.
 *
 * The Adreno driver of Windows on ARM machines answers CL_INVALID_DEVICE_TYPE
 * to CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, the type the start-up
 * used to create its context from. Squey 5.1.3 ended the process there
 * (BugSplat crashes 231 and 234), and walking past the error left those
 * machines without the GPU they have. The driver loaded here refuses the same
 * way, and offers a single GPU (see opencl_test_icd.cpp).
 *
 * When OCL_ICD_VENDORS names a library rather than a directory, ocl-icd loads
 * that library and nothing else, which keeps the drivers of the machine running
 * the test out of the way. ocl-icd is the loader of the Linux runtime only,
 * hence a test for Linux only.
 */
int main()
{
	PVCore::setenv("OCL_ICD_VENDORS", SQUEY_TEST_OPENCL_ICD, 1);
	// Unset rather than emptied: ocl-icd would take an empty name for a driver.
	unsetenv("OCL_ICD_FILENAMES");

	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	PV_VALID(platforms.size(), size_t{1});

	// Without the refusal, what follows would prove nothing.
	cl_int err = CL_SUCCESS;
	cl_context_properties properties[] = {
	    CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platforms.front()()), 0};
	const cl::Context refused(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, properties,
	                          nullptr, nullptr, &err);
	PV_VALID(err, CL_INVALID_DEVICE_TYPE);

	std::vector<std::string> devices;
	const auto record = [&devices](cl::Context&, cl::Device& device) {
		devices.push_back(device.getInfo<CL_DEVICE_NAME>());
	};

	const cl::Context accelerated = PVOpenCL::find_first_usable_context(true, record);
	PV_ASSERT_VALID(accelerated() != nullptr);
	PV_VALID(devices.size(), size_t{1});
	PV_VALID(devices.front(), std::string("Squey test GPU"));

	// The same driver has no CPU device to offer: the search for one comes back
	// empty-handed rather than with the GPU.
	devices.clear();
	const cl::Context software = PVOpenCL::find_first_usable_context(false, record);
	PV_ASSERT_VALID(software() == nullptr);
	PV_VALID(devices.size(), size_t{0});

	return 0;
}
