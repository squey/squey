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

#include <pvguiqt/PVAboutBoxDialog.h>

#include <pvparallelview/PVParallelView.h>

#include <QApplication>
#include <QLabel>

#include <string>

/**
 * Checks that the about box tells what the views are drawn on, not what a
 * search for OpenCL devices of its own turns up.
 *
 * The driver loaded here offers a GPU on which no command queue can be created
 * (see opencl_test_icd.cpp in the tests of libpvkernel), so the backend draws
 * on PortableCL instead, as Topencl_gpu_fallback checks. The about box used to
 * find that GPU by itself and announce hardware OpenCL support on it, next to
 * a status bar warning about the lack of GPU acceleration. The driver comes
 * first in the platform list for the reasons given in Topencl_gpu_fallback.
 */
int main(int argc, char** argv)
{
	PVCore::setenv("OCL_ICD_FILENAMES", SQUEY_TEST_OPENCL_ICD, 1);
	PVCore::setenv("OCL_ICD_PLATFORM_SORT", "none", 1);
	PVCore::setenv("FORCE_CPU", "0", 1);

	QApplication app(argc, argv); // argv carries "-platform offscreen"

	PVParallelView::common::RAII_backend_init backend_resources;
	PV_VALID(PVParallelView::common::is_gpu_accelerated(), false);

	const PVGuiQt::PVAboutBoxDialog about_box;
	const auto* software_info = about_box.findChild<QLabel*>("software_info");
	PV_ASSERT_VALID(software_info != nullptr);

	const std::string text = software_info->text().toStdString();
	PV_ASSERT_VALID(text.find("OpenCL™ support: software") != std::string::npos, "text", text);
	PV_ASSERT_VALID(text.find("Squey test GPU") == std::string::npos, "text", text);

	return 0;
}
