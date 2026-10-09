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

// What the drawing backends promise about the renderings still running when
// they are drained.

#include <pvparallelview/PVBCIBackendImage.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVBCIDrawingBackendOpenCL.h>
#include <pvparallelview/PVBCIDrawingBackendQPainter.h>

#include <pvkernel/core/PVUtils.h>
#include <pvkernel/core/squey_assert.h>

#include "bci_helpers.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

namespace
{

constexpr size_t BBITS = 10;
constexpr size_t ZONE_WIDTH = 512;
constexpr size_t CODE_COUNT = 10000;

/**
 * Asks for a rendering whose render_done keeps its thread for a while once it
 * is called, the way a thread preempted there does, and tells whether it has
 * returned.
 */
std::shared_ptr<std::atomic<bool>> render_slowly_done(PVParallelView::PVBCIDrawingBackend& backend,
                                                      PVParallelView::PVBCIBackendImage_p& image,
                                                      PVParallelView::PVBCICode<BBITS>* codes)
{
	auto returned = std::make_shared<std::atomic<bool>>(false);
	backend.render(image, 0, ZONE_WIDTH, reinterpret_cast<PVParallelView::PVBCICodeBase*>(codes),
	               CODE_COUNT, 1.0f, false, false, false, [returned]() {
		               std::this_thread::sleep_for(std::chrono::milliseconds(300));
		               *returned = true;
	               });
	return returned;
}

void check_wait_all(PVParallelView::PVBCIDrawingBackendAsync& backend,
                    char const* name,
                    PVParallelView::PVBCICode<BBITS>* codes)
{
	auto image = backend.create_image(ZONE_WIDTH, BBITS);
	auto returned = render_slowly_done(backend, image, codes);
	backend.wait_all();
	PV_ASSERT_VALID(returned->load(), "backend", name, "wait_all()", "returned before render_done");
}

} // namespace

int main()
{
	PVCore::setenv("FORCE_CPU", "1", 1);

	auto* codes = PVParallelView::PVBCICode<BBITS>::allocate_codes(CODE_COUNT);
	PVParallelView::PVBCIPatterns<BBITS>::init_codes_pattern(
	    codes, CODE_COUNT, PVParallelView::PVBCIPatterns<BBITS>::GRADIENT);

	auto& opencl = PVParallelView::PVBCIDrawingBackendOpenCL::get();
	PV_ASSERT_VALID(opencl.device_count() > 0, "device count", opencl.device_count());
	check_wait_all(opencl, "OpenCL", codes);
	check_wait_all(PVParallelView::PVBCIDrawingBackendQPainter::get(), "QPainter", codes);

	PVParallelView::PVBCICode<BBITS>::free_codes(codes);

	return 0;
}
