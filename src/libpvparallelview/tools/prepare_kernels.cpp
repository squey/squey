/* * MIT License
 *
 * © Squey, 2026
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

// Builds the rendering kernels for an application that may not build them itself:
// see PVBCIDrawingBackendOpenCL::prepare_prebuilt_kernels().

#include <pvparallelview/PVBCIDrawingBackendOpenCL.h>

#include <iostream>

int main(int argc, char** argv)
{
	if (argc != 2) {
		std::cerr << "Usage: " << argv[0] << " <directory>\n\n"
		          << "Builds the rendering kernels into <directory>, for the application to load\n"
		          << "them from there, through POCL_KERNEL_CACHE_PREBUILT_DIR.\n";
		return 2;
	}

	if (not PVParallelView::PVBCIDrawingBackendOpenCL::prepare_prebuilt_kernels(argv[1])) {
		std::cerr << "The rendering kernels could not all be built into " << argv[1] << "\n";
		return 1;
	}

	// The name of a PortableCL device carries the processor it builds for, which
	// selects the hash its kernels are filed under: kernels built for another one
	// than the application gets are never loaded.
	for (const auto& device : PVParallelView::PVBCIDrawingBackendOpenCL::get().opencl_devices()) {
		std::cout << "Built for " << device.name << "\n";
	}

	return 0;
}
