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

#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVBCIBackendImage.h>
#include <pvparallelview/PVBCIDrawingBackendOpenCL.h>

#include "bci_helpers.h"

#include <QDir>
#include <QImage>
#include <QString>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>

/**
 * Checks that a zone actually comes back drawn from the OpenCL backend.
 *
 * Topencl_cpu_device asserts that a device exists, which says nothing about
 * what the device produces: with a device present and the kernel building
 * without error, the application still showed black zones -- rendering goes
 * through an asynchronous path this test walks end to end, from BCI codes to
 * host pixels.
 *
 * The two renderings matter as much as the pixel check. PortableCL defers the
 * work-group specialisation of a kernel to the first enqueue that names
 * concrete dimensions, and writes the result to POCL_CACHE_DIR, so the first
 * rendering of a process with a cold cache is the one that pays for a compiler
 * run and the only one that can race with it. Pointing POCL_CACHE_DIR at an
 * empty directory reproduces that state on every run -- deleting the cache by
 * hand is how the black zones were reported in the first place -- and the
 * second rendering, cache now warm, gives the reference the first must match.
 */

static constexpr size_t BBITS = 10;
static constexpr size_t ZONE_WIDTH = 1024;
static constexpr size_t CODE_COUNT = 100000;

/**
 * Zone widths to draw. The kernel is enqueued over a whole number of
 * work-groups, so a width the work-group width does not divide leaves the last
 * one reaching past the zone; those work-items must not draw, and must still
 * reach every barrier. Widths that are neither a power of two nor a divisor of
 * the work-group width are there for exactly that.
 */
static constexpr size_t ZONE_WIDTHS[] = {1024, 1000, 513, 300, 64, 63, 1};

namespace
{

/**
 * Waits on the callback render() reports completion with, rather than on
 * wait_all(): finishing the queue would hide a callback that never fires, and
 * the callback is what the views repaint on.
 */
struct render_waiter {
	std::mutex mutex;
	std::condition_variable cv;
	bool done = false;
};

bool render_and_wait(PVParallelView::PVBCIDrawingBackendOpenCL& backend,
                     PVParallelView::PVBCIBackendImage_p& image,
                     PVParallelView::PVBCICodeBase* codes,
                     size_t n,
                     size_t width = ZONE_WIDTH)
{
	// Shared with the OpenCL callback thread so that giving up on the wait
	// below cannot leave that thread writing to a destroyed object.
	auto waiter = std::make_shared<render_waiter>();

	backend.render(image, 0, width, codes, n, 1.0f, false, [waiter]() {
		std::lock_guard<std::mutex> lock(waiter->mutex);
		waiter->done = true;
		waiter->cv.notify_all();
	});

	// Generous enough for a cold cache to run the compiler on a loaded machine,
	// short enough that two of these stay well inside SQUEY_TEST_TIMEOUT and
	// that a callback which never fires fails instead of hanging.
	std::unique_lock<std::mutex> lock(waiter->mutex);
	return waiter->cv.wait_for(lock, std::chrono::minutes(2), [&waiter]() { return waiter->done; });
}

size_t drawn_pixel_count(const QImage& image)
{
	size_t count = 0;

	for (int y = 0; y < image.height(); ++y) {
		const auto* line = reinterpret_cast<const uint32_t*>(image.constScanLine(y));

		for (int x = 0; x < image.width(); ++x) {
			// Opaque black is what an untouched image holds, whatever the alpha.
			if ((line[x] & 0x00ffffff) != 0) {
				++count;
			}
		}
	}

	return count;
}

/**
 * Cheap image identity, so that a rendering can be compared against the same
 * rendering from another build without keeping the pixels around.
 */
uint64_t checksum(const QImage& image)
{
	uint64_t sum = 1469598103934665603ull;

	for (int y = 0; y < image.height(); ++y) {
		const auto* line = reinterpret_cast<const uint32_t*>(image.constScanLine(y));

		for (int x = 0; x < image.width(); ++x) {
			sum = (sum ^ line[x]) * 1099511628211ull;
		}
	}

	return sum;
}

/**
 * Counts the kernel specialisations PortableCL has compiled and cached.
 *
 * It keeps one per local work size, under <cache>/<..>/<program>/DRAW/<shape>,
 * so the directories below a "DRAW" one are what a rendering had to wait for.
 */
size_t cached_kernel_count(const std::filesystem::path& cache_dir)
{
	size_t count = 0;
	std::error_code ec;

	for (std::filesystem::recursive_directory_iterator it(cache_dir, ec), end; it != end;
	     it.increment(ec)) {
		if (ec) {
			break;
		}

		if (it->is_directory(ec) && it->path().parent_path().filename() == "DRAW") {
			++count;
		}
	}

	return count;
}

} // namespace

int main()
{
	PVCore::setenv("FORCE_CPU", "1", 1);

	// Set before the backend builds the program, which is when PortableCL first
	// reads it. A directory of its own keeps a run from warming another's cache
	// when the suite runs in parallel.
	const QString cache_dir = PVCore::mkdtemp(QDir::tempPath() + "/squey_pocl_cache_XXXXXX");
	PV_ASSERT_VALID(not cache_dir.isEmpty(), "kernel cache directory", cache_dir.toStdString());
	PVCore::setenv("POCL_CACHE_DIR", qPrintable(cache_dir), 1);

	auto& backend = PVParallelView::PVBCIDrawingBackendOpenCL::get();
	PV_ASSERT_VALID(backend.device_count() > 0, "device count", backend.device_count());

	auto* codes = PVParallelView::PVBCICode<BBITS>::allocate_codes(CODE_COUNT);
	PVParallelView::PVBCIPatterns<BBITS>::init_codes_pattern(
	    codes, CODE_COUNT, PVParallelView::PVBCIPatterns<BBITS>::GRADIENT);

	PVParallelView::PVBCIBackendImage_p image = backend.create_image(ZONE_WIDTH, BBITS);

	// First rendering, cold kernel cache.
	PV_ASSERT_VALID(
	    render_and_wait(backend, image, reinterpret_cast<PVParallelView::PVBCICodeBase*>(codes),
	                    CODE_COUNT),
	    "rendering", "cold cache");

	// qimage() aliases the host buffer the next rendering writes to.
	const QImage cold = image->qimage().copy();
	const size_t cold_pixels = drawn_pixel_count(cold);

	// Second rendering, same codes, kernel now cached: what the application
	// gets from its second source onwards, and the reference for the first.
	PV_ASSERT_VALID(
	    render_and_wait(backend, image, reinterpret_cast<PVParallelView::PVBCICodeBase*>(codes),
	                    CODE_COUNT),
	    "rendering", "warm cache");

	const QImage warm = image->qimage().copy();
	const size_t warm_pixels = drawn_pixel_count(warm);

	std::cout << "drawn pixels: cold=" << cold_pixels << " warm=" << warm_pixels << std::endl;

	PV_ASSERT_VALID(warm_pixels > 0, "drawn pixels with a warm cache", warm_pixels);

	// The reported failure: a zone left black until the kernel cache exists.
	PV_ASSERT_VALID(cold_pixels > 0, "drawn pixels with a cold cache", cold_pixels);

	// Both renderings draw the same codes, so they owe the same image; a cold
	// cache that merely draws less is as wrong as one that draws nothing.
	PV_ASSERT_VALID(cold == warm, "cold drawn pixels", cold_pixels, "warm drawn pixels",
	                warm_pixels);

	// Every width the application can ask for has to come back drawn, not just
	// the one the two renderings above share. The checksums are printed so that
	// a change to how the work is split across work-groups can be checked to
	// leave the pixels alone.
	for (size_t width : ZONE_WIDTHS) {
		PVParallelView::PVBCIBackendImage_p zone = backend.create_image(width, BBITS);

		PV_ASSERT_VALID(render_and_wait(backend, zone,
		                                reinterpret_cast<PVParallelView::PVBCICodeBase*>(codes),
		                                CODE_COUNT, width),
		                "zone width", width);

		const QImage drawn = zone->qimage();
		const size_t pixels = drawn_pixel_count(drawn);

		std::cout << "width " << width << ": drawn=" << pixels << " checksum=0x" << std::hex
		          << checksum(drawn) << std::dec << std::endl;

		PV_ASSERT_VALID(pixels > 0, "zone width", width, "drawn pixels", pixels);
	}

	/* Zone widths must not each cost their own kernel: the work-group shape is
	 * rounded up to a power of two precisely so that they share one. Counting
	 * what landed in the cache is what tells the two apart -- the pixels are
	 * identical either way.
	 */
	const size_t kernels = cached_kernel_count(cache_dir.toStdString());
	const size_t widths = sizeof(ZONE_WIDTHS) / sizeof(ZONE_WIDTHS[0]);

	std::cout << "cached kernels: " << kernels << " for " << widths << " widths" << std::endl;

	// Zero means the cache was not laid out as expected -- another ICD, or a
	// PortableCL that moved things around -- and there is nothing to conclude.
	if (kernels > 0) {
		PV_ASSERT_VALID(kernels < widths, "cached kernels", kernels, "zone widths", widths);
	}

	PVParallelView::PVBCICode<BBITS>::free_codes(codes);

	std::error_code ec;
	std::filesystem::remove_all(cache_dir.toStdString(), ec);

	return 0;
}
