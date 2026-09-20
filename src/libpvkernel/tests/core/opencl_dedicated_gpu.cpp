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
 * Checks that the OpenCL start-up draws on the dedicated GPU of a hybrid laptop
 * rather than on the one built into its processor.
 *
 * Such a machine offers both, each on a platform of its own, in whatever order
 * the loader lists them: ocl-icd, as its directory reads. On the one it was
 * measured on, the built-in GPU drew the parallel view 2.8 times slower than
 * the dedicated one, an Iris Xe against an RTX 3060. The driver loaded here
 * lists its built-in GPU first (see opencl_test_icd.cpp), and the platform
 * sorting of ocl-icd is turned off so that the order holds.
 */
int main()
{
	PVCore::setenv("OCL_ICD_VENDORS", SQUEY_TEST_OPENCL_ICD, 1);
	// Unset rather than emptied: ocl-icd would take an empty name for a driver.
	unsetenv("OCL_ICD_FILENAMES");
	PVCore::setenv("OCL_ICD_PLATFORM_SORT", "none", 1);

	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	PV_VALID(platforms.size(), size_t{2});

	// Without the built-in GPU coming first, what follows would prove nothing.
	std::vector<cl::Device> first_devices;
	platforms.front().getDevices(CL_DEVICE_TYPE_ALL, &first_devices);
	PV_VALID(first_devices.size(), size_t{1});
	const std::string first_device = first_devices.front().getInfo<CL_DEVICE_NAME>();
	PV_VALID(first_device, std::string("Squey test integrated GPU"));

	std::vector<std::string> devices;
	const auto record = [&devices](cl::Context&, cl::Device& device) {
		devices.push_back(device.getInfo<CL_DEVICE_NAME>());
	};

	const cl::Context accelerated = PVOpenCL::find_first_usable_context(true, record);
	PV_ASSERT_VALID(accelerated() != nullptr);
	PV_VALID(devices.size(), size_t{1});
	PV_VALID(devices.front(), std::string("Squey test dedicated GPU"));

	return 0;
}
