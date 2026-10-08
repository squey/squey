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

// The scatter thumbnails gallery renders off the scaled columns directly
// instead of going through the zoomed zone tree the real scatter view uses (a
// tree costs a 1024x1024 grid of quadtrees per axis pair, which a gallery
// showing every pair cannot afford). That shortcut has to land on exactly the
// same pixels, and the mapping is easy to get wrong in a way no crash reveals:
// PVScatterView renders y1 increasing rightwards, then flips the image
// horizontally before drawing it (PVScatterView::RenderedImage::swap), so a
// thumbnail must invert x and leave y alone.
//
// This test renders one pair both ways and checks which of the four flips of
// the reference lines up with the thumbnail. A mirrored thumbnail scores near
// zero on the right orientation, so the oracle catches it however subtle it
// looks on screen.

#include <squey/PVScaled.h>
#include <squey/PVView.h>

#include <pvkernel/core/squey_assert.h>

#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVScatterThumbnail.h>
#include <pvparallelview/PVScatterViewDataImpl.h>
#include <pvparallelview/PVScatterViewImage.h>
#include <pvparallelview/PVViewRenderingContext.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/common.h>

#include <QImage>

#include <atomic>
#include <iostream>

#include "common.h"

static constexpr const char* filename = TEST_FOLDER "/picviz/heat_line.csv";
static constexpr const char* fileformat = TEST_FOLDER "/picviz/heat_line.csv.format";

//! Pixels both images draw on, with @a ref flipped as asked.
static size_t overlap(QImage const& mine, QImage const& ref, bool flip_x, bool flip_y)
{
	const int size = mine.width();
	size_t common = 0;
	for (int y = 0; y < size; ++y) {
		for (int x = 0; x < size; ++x) {
			const int rx = flip_x ? size - 1 - x : x;
			const int ry = flip_y ? size - 1 - y : y;
			common += qAlpha(mine.pixel(x, y)) != 0 and qAlpha(ref.pixel(rx, ry)) != 0;
		}
	}
	return common;
}

int main()
{
	PVParallelView::common::RAII_backend_init resources;
	TestEnv env(filename, fileformat);

	PVParallelView::PVViewRenderingContext* context = env.get_rendering_context();
	Squey::PVView& view = *context->lib_view();

	const PVCol x_col(0);
	const PVCol y_col(1);
	// Same resolution as PVScatterViewImage, so the two renders compare pixel
	// to pixel with no resampling in between.
	constexpr int size = PVParallelView::PVScatterViewImage::image_width;

	// The gallery's render.
	const std::atomic<bool> not_cancelled{false};
	PVParallelView::PVScatterThumbnailImages thumbnail;
	PV_ASSERT_VALID(PVParallelView::PVScatterThumbnail::render(view, x_col, y_col, size, thumbnail,
	                                                           not_cancelled));

	// The real scatter render on the same pair, called synchronously.
	auto retainer = context->acquire_zoomed_zone(PVZoneID{x_col, y_col});
	auto const& zzt = context->get_zones_manager().get_zoom_zone_tree(PVZoneID{x_col, y_col});

	PVParallelView::PVScatterViewImage reference;
	PVParallelView::PVScatterViewDataInterface::ProcessParams params(
	    zzt, view.get_output_layer_color_buffer());
	// shift = (32 - PARALLELVIEW_ZT_BBITS) - zoom, and the image is 2048 wide,
	// so zoom = 1 maps the whole 2^32 scene onto it exactly once.
	params.set_params(0, 0xFFFFFFFFULL, 0, 0xFFFFFFFFULL, 1, 1.0);

	PVParallelView::PVScatterViewDataImpl().process_bg(params, reference);
	reference.convert_image_from_hsv_to_rgb();

	QImage const& mine = thumbnail.bg;
	QImage const& ref = reference.get_rgb_image();

	const size_t expected = overlap(mine, ref, true, false); // the horizontal flip
	const size_t identity = overlap(mine, ref, false, false);
	const size_t vertical = overlap(mine, ref, false, true);
	const size_t both = overlap(mine, ref, true, true);

	std::cout << "overlap flipH=" << expected << " identity=" << identity
	          << " flipV=" << vertical << " flipHV=" << both << std::endl;

	// The zoomed zone tree returns one row per quadtree bucket, so it draws a
	// subset of what the thumbnail does: every reference pixel must be covered,
	// but not the other way round.
	size_t reference_pixels = 0;
	for (int y = 0; y < size; ++y) {
		for (int x = 0; x < size; ++x) {
			reference_pixels += qAlpha(ref.pixel(x, y)) != 0;
		}
	}

	PV_ASSERT_VALID(reference_pixels > 0, "reference_pixels", reference_pixels);
	PV_ASSERT_VALID(expected == reference_pixels, "matched", expected, "reference_pixels",
	                reference_pixels);

	// A mirrored render would score highest on one of these instead.
	PV_ASSERT_VALID(expected > identity, "flipH", expected, "identity", identity);
	PV_ASSERT_VALID(expected > vertical, "flipH", expected, "flipV", vertical);
	PV_ASSERT_VALID(expected > both, "flipH", expected, "flipHV", both);

	return 0;
}
