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

#include <pvparallelview/PVScatterThumbnail.h>

#include <squey/PVScaled.h>
#include <squey/PVSelection.h>
#include <squey/PVView.h>

#include <pvkernel/core/PVHSVColor.h>

#include <algorithm>

#include <cassert>
#include <cmath>
#include <limits>
#include <vector>

namespace
{

/**
 * How often the cancellation flag is polled, in rows.
 *
 * The loop body is a couple of shifts and a store, so reading an atomic every
 * iteration would dominate it. A row block is short enough that a cancelled
 * render still stops in well under a millisecond.
 */
constexpr PVRow cancellation_poll_block = 1u << 16;

/**
 * Scaled values are mapped into [0, 1] before being accumulated: summing
 * products of raw uint32_t values overflows 64 bits well before the row counts
 * this code is meant to handle.
 */
constexpr double value_scale = 1. / double(std::numeric_limits<uint32_t>::max());

/**
 * Rows to step over so that at most @a threshold of them are visited.
 */
PVRow sampling_stride(PVRow row_count, PVRow threshold)
{
	if (row_count <= threshold) {
		return 1;
	}
	return (row_count + threshold - 1) / threshold;
}

/**
 * Project a scaled pair onto a thumbnail pixel, reproducing what a scatter
 * view ends up showing.
 *
 * PVScatterView renders y1 (the horizontal axis) increasing rightwards into
 * its image, then flips that image horizontally before drawing it
 * (PVScatterView::RenderedImage::swap). Inverting x here is the same
 * operation done once, and matches PVCore::invert_scaling_value(), which the
 * scatter view applies to go from scene coordinates back to scaled values.
 *
 * The vertical axis is left as is: row 0 of the image is the top of the view,
 * exactly as in the scatter image.
 *
 * The fixed-point multiply maps [0, 2^32) onto [0, size) for any size, which
 * a right shift would only do for powers of two -- and the gallery's size
 * control is a slider.
 */
inline size_t pixel_index(uint32_t x, uint32_t y, int size)
{
	const uint32_t col = uint32_t((uint64_t(~x) * uint64_t(size)) >> 32);
	const uint32_t row = uint32_t((uint64_t(y) * uint64_t(size)) >> 32);
	assert(col < uint32_t(size) and row < uint32_t(size));
	return size_t(row) * size_t(size) + col;
}

/**
 * Convert an HSV buffer to the ARGB image the gallery draws.
 */
QImage to_image(std::vector<PVCore::PVHSVColor> const& hsv, int size)
{
	QImage image(size, size, QImage::Format_ARGB32);
	PVCore::PVHSVColor::to_rgba(hsv.data(), image);
	return image;
}

} // namespace

bool PVParallelView::PVScatterThumbnail::render(Squey::PVView const& view,
                                                PVCol x_col,
                                                PVCol y_col,
                                                int size,
                                                PVScatterThumbnailImages& out,
                                                std::atomic<bool> const& cancelled)
{
	assert(size > 0);

	const PVRow row_count = view.get_row_count();
	if (row_count == 0) {
		return false;
	}

	const auto& scaled = view.get_parent<Squey::PVScaled>();
	const uint32_t* const x = scaled.get_column_pointer(x_col);
	const uint32_t* const y = scaled.get_column_pointer(y_col);
	const PVCore::PVHSVColor* const colors = view.get_output_layer_color_buffer();
	const Squey::PVSelection& sel = view.get_real_output_selection();
	// Nothing selected means an empty selection layer, so the per-row lookup
	// and its second buffer can go entirely -- which is the state the gallery
	// opens in.
	const bool has_selection = not sel.is_empty();

	const PVRow stride = sampling_stride(row_count, row_sampling_threshold);
	const size_t pixels = size_t(size) * size_t(size);

	std::vector<PVCore::PVHSVColor> hsv_bg(pixels, HSV_COLOR_TRANSPARENT);
	std::vector<PVCore::PVHSVColor> hsv_sel(has_selection ? pixels : 0, HSV_COLOR_TRANSPARENT);

	const size_t block_span = size_t(cancellation_poll_block) * size_t(stride);
	for (size_t block = 0; block < row_count; block += block_span) {
		if (cancelled) {
			return false;
		}
		const PVRow block_end = PVRow(std::min<size_t>(row_count, block + block_span));

		for (PVRow i = PVRow(block); i < block_end; i += stride) {
			const size_t p = pixel_index(x[i], y[i], size);

			// First row reaching a pixel owns it, as in the real scatter render
			// (PVZoomedZoneTree.cpp). Overwriting instead would make the image
			// depend on row order in a way the scatter view's does not.
			if (hsv_bg[p] == HSV_COLOR_TRANSPARENT) {
				hsv_bg[p] = colors[i];
			}
			if (has_selection and hsv_sel[p] == HSV_COLOR_TRANSPARENT and sel.get_line_fast(i)) {
				hsv_sel[p] = colors[i];
			}
		}
	}

	if (cancelled) {
		return false;
	}

	out.bg = to_image(hsv_bg, size);
	out.sel = has_selection ? to_image(hsv_sel, size) : QImage();
	out.sampled = stride > 1;

	return true;
}

PVRow PVParallelView::PVScatterThumbnail::correlation_sample_size(PVRow row_count,
                                                                  size_t pair_count)
{
	if (pair_count == 0) {
		return std::min(row_count, correlation_sampling_threshold);
	}

	// Two columns read per pair.
	const size_t per_pair = correlation_sample_budget / (2 * pair_count);
	const PVRow capped = PVRow(
	    std::clamp<size_t>(per_pair, correlation_min_sample, correlation_sampling_threshold));

	return std::min(row_count, capped);
}

PVParallelView::PVScatterThumbnail::ColumnMoments
PVParallelView::PVScatterThumbnail::column_moments(Squey::PVView const& view,
                                                   PVCol col,
                                                   PVRow sample_size)
{
	ColumnMoments moments;

	const PVRow row_count = view.get_row_count();
	const uint32_t* const v = view.get_parent<Squey::PVScaled>().get_column_pointer(col);
	const PVRow stride = sampling_stride(row_count, sample_size);

	for (PVRow i = 0; i < row_count; i += stride) {
		const double vi = double(v[i]) * value_scale;
		moments.sum += vi;
		moments.sum_sq += vi * vi;
		++moments.count;
	}

	return moments;
}

double PVParallelView::PVScatterThumbnail::correlation(Squey::PVView const& view,
                                                       PVCol x_col,
                                                       PVCol y_col,
                                                       ColumnMoments const& x_moments,
                                                       ColumnMoments const& y_moments,
                                                       PVRow sample_size)
{
	const PVRow row_count = view.get_row_count();
	const size_t n = x_moments.count;
	if (row_count < 2 or n < 2 or y_moments.count != n) {
		return 0.;
	}

	const auto& scaled = view.get_parent<Squey::PVScaled>();
	const uint32_t* const x = scaled.get_column_pointer(x_col);
	const uint32_t* const y = scaled.get_column_pointer(y_col);

	const PVRow stride = sampling_stride(row_count, sample_size);

	// Only the cross term is per-pair; the per-column sums were walked once
	// each by column_moments().
	double sum_xy = 0.;
	for (PVRow i = 0; i < row_count; i += stride) {
		sum_xy += (double(x[i]) * value_scale) * (double(y[i]) * value_scale);
	}

	const double cov = sum_xy - x_moments.sum * y_moments.sum / double(n);
	const double var_x = x_moments.sum_sq - x_moments.sum * x_moments.sum / double(n);
	const double var_y = y_moments.sum_sq - y_moments.sum * y_moments.sum / double(n);

	// A constant column has no variance and so no correlation to report; it
	// would otherwise divide by zero.
	if (var_x <= 0. or var_y <= 0.) {
		return 0.;
	}

	// Rounding can push a perfect correlation just past 1, which would sort
	// ahead of a genuine 1 and read as a bug.
	return std::min(1., std::abs(cov / std::sqrt(var_x * var_y)));
}
