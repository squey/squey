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

#include <pvkernel/core/PVKernelSettings.h>
#include <pvkernel/core/PVSettings.h>
#include <pvkernel/core/PVUtils.h>
#include <pvkernel/core/squey_assert.h>
#include <pvkernel/opencl/common.h>

#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

/**
 * Checks that the settings name the OpenCL platform that draws the parallel
 * coordinates, among those the settings dialog lists with their devices.
 *
 * The driver loaded here stands for the two GPUs of a hybrid laptop, the one
 * built into the processor listed first (see opencl_test_icd.cpp). Left to
 * itself, the start-up would draw on the dedicated one.
 */
int main()
{
	using kind_t = PVOpenCL::lines_platform_t::kind_t;

	PVCore::setenv("OCL_ICD_VENDORS", SQUEY_TEST_OPENCL_ICD, 1);
	// Unset rather than emptied: ocl-icd would take an empty name for a driver.
	unsetenv("OCL_ICD_FILENAMES");
	PVCore::setenv("OCL_ICD_PLATFORM_SORT", "none", 1);
	unsetenv("FORCE_CPU");

	const std::vector<PVOpenCL::platform_t> platforms = PVOpenCL::platforms();
	PV_VALID(platforms.size(), size_t{2});
	PV_VALID(platforms[0].name, std::string("Squey test integrated platform"));
	PV_VALID(platforms[0].accelerated, true);
	PV_VALID(platforms[0].devices.size(), size_t{1});
	PV_VALID(platforms[0].devices.front(), std::string("Squey test integrated GPU"));
	PV_VALID(platforms[1].name, std::string("Squey test dedicated platform"));

	const auto descriptor =
	    PVCore::PVSettings::get().descriptor(PVCore::settings::lines_platform.key());
	PV_ASSERT_VALID(descriptor and descriptor->machine_choices);
	const std::vector<PVCore::PVSettingChoice> choices = descriptor->machine_choices();
	PV_VALID(choices.size(), size_t{4});
	PV_VALID(choices[0].value.toStdString(), std::string("auto"));
	PV_VALID(choices[1].value.toStdString(), std::string("gpu:Squey test integrated platform"));
	PV_VALID(
	    choices[1].label.toStdString(),
	    std::string("Squey test integrated platform (graphics card: Squey test integrated GPU)"));
	PV_VALID(choices[2].value.toStdString(), std::string("gpu:Squey test dedicated platform"));
	PV_VALID(choices[3].value.toStdString(), std::string("qt"));

	const auto lines_platform = [](const char* value) {
		PV_ASSERT_VALID(PVCore::settings::lines_platform.set_value(value), "value", value);
		return PVOpenCL::lines_platform();
	};

	std::vector<std::string> devices;
	const auto record = [&devices](cl::Context&, cl::Device& device) {
		devices.push_back(device.getInfo<CL_DEVICE_NAME>());
	};

	const PVOpenCL::lines_platform_t integrated =
	    lines_platform("gpu:Squey test integrated platform");
	PV_ASSERT_VALID(integrated.kind == kind_t::accelerated);
	PV_VALID(integrated.name, std::string("Squey test integrated platform"));
	const cl::Context context = PVOpenCL::find_first_usable_context(true, record, integrated.name);
	PV_ASSERT_VALID(context() != nullptr);
	PV_VALID(devices.size(), size_t{1});
	PV_VALID(devices.front(), std::string("Squey test integrated GPU"));

	// A platform the machine does not offer has no device to draw on.
	devices.clear();
	PV_ASSERT_VALID(PVOpenCL::find_first_usable_context(true, record, "Absent platform")() ==
	                nullptr);
	PV_VALID(devices.size(), size_t{0});

	// The test suite keeps the rendering on a CPU device, whichever GPU is named.
	PVCore::setenv("FORCE_CPU", "1", 1);
	const PVOpenCL::lines_platform_t forced = PVOpenCL::lines_platform();
	PV_ASSERT_VALID(forced.kind == kind_t::processor and forced.name.empty());
	unsetenv("FORCE_CPU");

	const PVOpenCL::lines_platform_t processor =
	    lines_platform("cpu:Squey test dedicated platform");
	PV_ASSERT_VALID(processor.kind == kind_t::processor);
	PV_VALID(processor.name, std::string("Squey test dedicated platform"));
	const PVOpenCL::lines_platform_t any_processor = lines_platform("cpu");
	PV_ASSERT_VALID(any_processor.kind == kind_t::processor and any_processor.name.empty());
	PV_ASSERT_VALID(lines_platform("qt").kind == kind_t::qt);
	PV_ASSERT_VALID(lines_platform("auto").kind == kind_t::automatic);
	// That of a later version, say, is left to the start-up.
	PV_ASSERT_VALID(lines_platform("npu:Squey test dedicated platform").kind == kind_t::automatic);

	return 0;
}
