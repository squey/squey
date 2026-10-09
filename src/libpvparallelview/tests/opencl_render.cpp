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
#include <pvparallelview/PVBCIDrawingBackendQPainter.h>

#include "bci_helpers.h"

#include <QDir>
#include <QImage>
#include <QString>

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

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

bool render_and_wait(PVParallelView::PVBCIDrawingBackendAsync& backend,
                     PVParallelView::PVBCIBackendImage_p& image,
                     PVParallelView::PVBCICodeBase* codes,
                     size_t n,
                     size_t width = ZONE_WIDTH,
                     bool density = false,
                     bool antialiased = false)
{
	// Shared with the OpenCL callback thread so that giving up on the wait
	// below cannot leave that thread writing to a destroyed object.
	auto waiter = std::make_shared<render_waiter>();

	backend.render(image, 0, width, codes, n, 1.0f, false, density, antialiased, [waiter]() {
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

using pixels_t = std::set<std::pair<int, int>>;

//! Where a rendering drew, however faintly.
pixels_t drawn_at(const QImage& image)
{
	pixels_t drawn;

	for (int y = 0; y < image.height(); ++y) {
		const auto* line = reinterpret_cast<const uint32_t*>(image.constScanLine(y));

		for (int x = 0; x < image.width(); ++x) {
			if (qAlpha(line[x]) != 0) {
				drawn.emplace(x, y);
			}
		}
	}

	return drawn;
}

uint32_t pixel_at(const QImage& image, std::pair<int, int> const& at)
{
	return reinterpret_cast<const uint32_t*>(image.constScanLine(at.second))[at.first];
}

PVParallelView::PVBCICode<BBITS>
line_code(PVRow row, uint8_t opacity, uint32_t left, uint32_t right, uint8_t color)
{
	PVParallelView::PVBCICode<BBITS> code;
	code.int_v = 0;
	code.s.idx = row;
	code.set_opacity(opacity);
	code.s.l = left;
	code.s.r = right;
	code.s.color = color;
	return code;
}

QImage render_lines(PVParallelView::PVBCIDrawingBackendAsync& backend,
                    std::vector<PVParallelView::PVBCICode<BBITS>> codes,
                    bool density,
                    bool antialiased = false,
                    size_t width = ZONE_WIDTH)
{
	PVParallelView::PVBCIBackendImage_p image = backend.create_image(width, BBITS);

	PV_ASSERT_VALID(render_and_wait(backend, image,
	                                reinterpret_cast<PVParallelView::PVBCICodeBase*>(codes.data()),
	                                codes.size(), width, density, antialiased),
	                "rendering", "lines");

	return image->qimage().copy();
}

/**
 * Drawn by density, each line carries its own opacity, and where lines cross,
 * the most opaque one shows rather than the one of the lowest row. Black lines,
 * the zombies, still stay behind every other, however opaque.
 *
 * Both backends owe the same, and the views fall back on QPainter when there is
 * no OpenCL device: they are checked alike, on the opacity of what they draw.
 * The OpenCL image is a premultiplied one, which its colours are checked for.
 */
void check_density(PVParallelView::PVBCIDrawingBackendAsync& backend,
                   std::string const& name,
                   bool premultiplied)
{
	// A faint line of a low row, rising across the zone, and level ones crossing
	// it: two diagonals can pass through each other without sharing a pixel.
	const auto faint = line_code(0, 40, 0, 1023, 10);
	const auto opaque = line_code(1 << 20, 200, 300, 300, 100);
	const auto zombie = line_code(0, 255, 700, 700, HSV_COLOR_BLACK.h());

	const QImage faint_alone = render_lines(backend, {faint}, true);
	const pixels_t faint_at = drawn_at(faint_alone);
	PV_ASSERT_VALID(not faint_at.empty(), "backend", name, "faint line", "not drawn");

	for (auto const& at : faint_at) {
		PV_ASSERT_VALID(qAlpha(pixel_at(faint_alone, at)) == 40, "backend", name, "x", at.first,
		                "y", at.second, "opacity", qAlpha(pixel_at(faint_alone, at)));
	}

	if (premultiplied) {
		// Drawn opaque, the same line gives the colour the opacity scales.
		const QImage faint_opaque = render_lines(backend, {faint}, false);

		for (auto const& at : faint_at) {
			const QRgb opaque_pixel = pixel_at(faint_opaque, at);
			const QRgb expected = qPremultiply(
			    qRgba(qRed(opaque_pixel), qGreen(opaque_pixel), qBlue(opaque_pixel), 40));
			const QRgb drawn = pixel_at(faint_alone, at);
			const auto close = [](int a, int b) { return std::abs(a - b) <= 1; };

			PV_ASSERT_VALID(close(qRed(drawn), qRed(expected)) and
			                    close(qGreen(drawn), qGreen(expected)) and
			                    close(qBlue(drawn), qBlue(expected)),
			                "backend", name, "x", at.first, "y", at.second, "drawn", drawn,
			                "premultiplied", expected);
		}
	}

	const pixels_t opaque_at = drawn_at(render_lines(backend, {opaque}, true));
	const pixels_t zombie_at = drawn_at(render_lines(backend, {zombie}, true));

	const QImage crossing = render_lines(backend, {faint, opaque}, true);
	const QImage behind = render_lines(backend, {zombie, faint}, true);
	size_t crossed_opaque = 0;
	size_t crossed_zombie = 0;

	for (auto const& at : faint_at) {
		if (opaque_at.count(at) != 0) {
			++crossed_opaque;
			PV_ASSERT_VALID(qAlpha(pixel_at(crossing, at)) == 200, "backend", name, "x", at.first,
			                "y", at.second, "opacity where it crosses the opaque line",
			                qAlpha(pixel_at(crossing, at)));
		}
		if (zombie_at.count(at) != 0) {
			++crossed_zombie;
			PV_ASSERT_VALID(qAlpha(pixel_at(behind, at)) == 40, "backend", name, "x", at.first,
			                "y", at.second, "opacity where it crosses the zombie",
			                qAlpha(pixel_at(behind, at)));
		}
	}

	PV_ASSERT_VALID(crossed_opaque > 0, "backend", name, "the opaque line", "never crossed");
	PV_ASSERT_VALID(crossed_zombie > 0, "backend", name, "the zombie", "never crossed");

	std::cout << name << " by density: " << crossed_opaque << " pixels where the opaque line crosses, "
	          << crossed_zombie << " where the zombie does" << std::endl;
}

/**
 * Antialiased, a line covers the pixels it passes near in part, by how close it
 * passes to their centre along its minor axis. Its opacity is shared out between
 * the pixels it passes between: each column of a flat line, and each row of a
 * steep one, holds the whole of it. Where lines meet, a pixel keeps the line
 * covering it the most, black lines staying behind the others.
 */
void check_antialiasing(PVParallelView::PVBCIDrawingBackendAsync& backend)
{
	const auto alpha_at = [](QImage const& image, int x, int y) {
		return qAlpha(pixel_at(image, {x, y}));
	};

	// On a whole row, a level line covers that row entirely, and nothing else.
	const auto level = line_code(0, 255, 300, 300, 100);
	const QImage level_aliased = render_lines(backend, {level}, false);
	const QImage level_alone = render_lines(backend, {level}, false, true);
	PV_ASSERT_VALID(level_alone == level_aliased, "level line on a whole row",
	                "differs from the aliased one");

	const auto flat = line_code(0, 255, 300, 301, 100);
	const QImage flat_alone = render_lines(backend, {flat}, false, true);
	size_t partial = 0;

	for (int x = 0; x < static_cast<int>(ZONE_WIDTH); ++x) {
		int column = 0;

		for (int y = 0; y < flat_alone.height(); ++y) {
			const int alpha = alpha_at(flat_alone, x, y);
			PV_ASSERT_VALID(alpha == 0 or y == 300 or y == 301, "flat line x", x, "y", y,
			                "opacity", alpha);
			partial += alpha != 0 and alpha != 255;
			column += alpha;
		}

		PV_ASSERT_VALID(std::abs(column - 255) <= 1, "flat line column", x, "opacity", column);
	}

	PV_ASSERT_VALID(partial > 0, "flat line", "covers no pixel in part");

	constexpr size_t steep_width = 64;
	const auto steep = line_code(0, 255, 0, 1023, 100);
	const QImage steep_alone = render_lines(backend, {steep}, false, true, steep_width);

	// Short of the ends, where the line leaves the zone with part of a row.
	for (int y = 32; y < 1023 - 32; ++y) {
		int row = 0;

		for (int x = 0; x < static_cast<int>(steep_width); ++x) {
			row += alpha_at(steep_alone, x, y);
		}

		PV_ASSERT_VALID(std::abs(row - 255) <= 2, "steep line row", y, "opacity", row);
	}

	// Drawn by density as well, the coverage scales the opacity of the line.
	const QImage faint_alone = render_lines(backend, {line_code(0, 40, 300, 301, 100)}, true, true);

	for (int x = 0; x < static_cast<int>(ZONE_WIDTH); ++x) {
		for (int y : {300, 301}) {
			const double expected = 40. * alpha_at(flat_alone, x, y) / 255.;
			PV_ASSERT_VALID(std::abs(alpha_at(faint_alone, x, y) - expected) <= 1., "faint line x",
			                x, "y", y, "opacity", alpha_at(faint_alone, x, y), "expected",
			                expected);
		}
	}

	// The line of the lowest row wins where both cover a pixel as much.
	const auto diagonal = line_code(1 << 20, 255, 0, 1023, 50);
	const auto zombie = line_code(0, 255, 700, 700, HSV_COLOR_BLACK.h());
	const QImage diagonal_alone = render_lines(backend, {diagonal}, false, true);
	const QImage zombie_alone = render_lines(backend, {zombie}, false, true);
	const QImage met = render_lines(backend, {level, diagonal}, false, true);
	const QImage behind = render_lines(backend, {zombie, diagonal}, false, true);
	size_t shared_with_level = 0;
	size_t shared_with_zombie = 0;

	for (int y = 0; y < met.height(); ++y) {
		for (int x = 0; x < met.width(); ++x) {
			const QRgb by_diagonal = pixel_at(diagonal_alone, {x, y});
			const QRgb by_level = pixel_at(level_alone, {x, y});
			const QRgb by_zombie = pixel_at(zombie_alone, {x, y});

			if (qAlpha(by_diagonal) == 0) {
				continue;
			}

			if (qAlpha(by_level) != 0) {
				++shared_with_level;
				const QRgb expected = qAlpha(by_diagonal) > qAlpha(by_level) ? by_diagonal : by_level;
				PV_ASSERT_VALID(pixel_at(met, {x, y}) == expected, "x", x, "y", y, "drawn",
				                pixel_at(met, {x, y}), "expected", expected);
			}

			if (qAlpha(by_zombie) != 0) {
				++shared_with_zombie;
				PV_ASSERT_VALID(pixel_at(behind, {x, y}) == by_diagonal, "x", x, "y", y, "drawn",
				                pixel_at(behind, {x, y}), "over the zombie", by_diagonal);
			}
		}
	}

	PV_ASSERT_VALID(shared_with_level > 0, "the diagonal", "never meets the level line");
	PV_ASSERT_VALID(shared_with_zombie > 0, "the diagonal", "never meets the zombie");

	std::cout << "OpenCL antialiased: " << partial << " pixels of the flat line covered in part, "
	          << shared_with_level << " shared with the level line, " << shared_with_zombie
	          << " with the zombie" << std::endl;
}

/**
 * QPainter antialiases on its own terms, blending where lines meet: only check
 * that it covers pixels in part, around the rows the line passes between.
 */
void check_antialiasing_qpainter()
{
	auto& backend = PVParallelView::PVBCIDrawingBackendQPainter::get();
	const QImage flat_alone =
	    render_lines(backend, {line_code(0, 255, 300, 301, 100)}, false, true);
	size_t partial = 0;

	for (auto const& at : drawn_at(flat_alone)) {
		PV_ASSERT_VALID(at.second >= 299 and at.second <= 302, "QPainter flat line x",
		                at.first, "y", at.second);
		partial += qAlpha(pixel_at(flat_alone, at)) != 255;
	}

	PV_ASSERT_VALID(partial > 0, "QPainter flat line", "covers no pixel in part");
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
	 * rounded up to a power of two precisely so that they share one. On the
	 * processor, a work-group draws a few columns whatever the zone width (see
	 * PARALLELVIEW_POCL_CPU_LOCAL_MEM_SIZE), so the widths below share a single
	 * shape, all but the one-pixel zone. Counting what landed in the cache is what
	 * tells them apart -- the pixels are identical either way.
	 */
	const size_t kernels = cached_kernel_count(cache_dir.toStdString());
	const size_t widths = sizeof(ZONE_WIDTHS) / sizeof(ZONE_WIDTHS[0]);

	std::cout << "cached kernels: " << kernels << " for " << widths << " widths" << std::endl;

	// Zero means the cache was not laid out as expected -- another ICD, or a
	// PortableCL that moved things around -- and there is nothing to conclude.
	if (kernels > 0) {
		PV_ASSERT_VALID(kernels <= 2, "cached kernels", kernels, "zone widths", widths);
	}

	check_density(backend, "OpenCL", true);
	check_density(PVParallelView::PVBCIDrawingBackendQPainter::get(), "QPainter", false);

	check_antialiasing(backend);
	check_antialiasing_qpainter();

	/* A rendering reports its end from inside its job, which is not over yet. Wait for
	 * the jobs while the threads that run them are still there: on Windows, the statics
	 * of a DLL are destroyed once every other thread is gone, and the destructor of the
	 * QPainter backend would wait for the job of the last rendering forever.
	 */
	PVParallelView::PVBCIDrawingBackendQPainter::get().wait_all();
	backend.wait_all();

	PVParallelView::PVBCICode<BBITS>::free_codes(codes);

	std::error_code ec;
	std::filesystem::remove_all(cache_dir.toStdString(), ec);

	return 0;
}
