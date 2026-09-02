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

#ifndef PVPARALLELVIEW_PVSCATTERTHUMBNAIL_H
#define PVPARALLELVIEW_PVSCATTERTHUMBNAIL_H

#include <pvbase/types.h>

#include <QImage>

#include <atomic>
#include <cstdint>

namespace Squey
{
class PVView;
} // namespace Squey

namespace PVParallelView
{

/**
 * The two layers of one thumbnail, drawn the way PVScatterView draws its own:
 * the background at low opacity, the selection at full opacity on top of it
 * (see PVScatterView::drawBackground).
 *
 * They are kept apart rather than composited here because they are invalidated
 * on different events: moving the selection leaves @a bg untouched.
 */
struct PVScatterThumbnailImages {
	QImage bg;
	QImage sel;
	//! Set when the render walked only a fraction of the rows (see row_sampling_threshold).
	bool sampled = false;

	bool is_null() const { return bg.isNull(); }
};

/**
 * Renders one scatter thumbnail straight from the scaled columns.
 *
 * This deliberately bypasses the PVScatterViewBackend / PVZoomedZoneTree
 * pipeline the real scatter view uses: a zoomed zone tree allocates a
 * 1024x1024 grid of quadtrees per axis pair, which a gallery showing every
 * pair at once cannot afford. A thumbnail needs no zooming and no
 * hit-testing, only a fixed low-resolution projection, so scaled values --
 * already one uint32_t per row and per column -- can be shifted straight into
 * pixel coordinates.
 *
 * The cost is O(rows) per thumbnail with no per-pair allocation beyond the
 * image itself, which is what makes rendering the visible thumbnails of a
 * large gallery practical.
 *
 * Nothing here touches the widget layer, so a render can run on any thread
 * (and be unit-tested without a QApplication). Callers are expected to
 * parallelise *across* thumbnails: a single render writes its buffer
 * unsynchronised and must stay on one thread.
 */
class PVScatterThumbnail
{
  public:
	/**
	 * Row count above which the render walks a strided sample instead of every
	 * row.
	 *
	 * A thumbnail holds a few tens of thousands of pixels, and the first row
	 * reaching a pixel owns it, so past this point extra rows almost never
	 * change what is drawn -- while they do keep costing a full pass. Below the
	 * threshold nothing is skipped: an aperçu that silently drops a small
	 * cluster is worse than a slower one.
	 */
	constexpr static PVRow row_sampling_threshold = 20 * 1000 * 1000;

	/**
	 * Render the (@a x_col, @a y_col) pair into @a out.
	 *
	 * @param size edge length in pixels; any value works, a scaled value is
	 *             mapped to a pixel by a fixed-point multiply rather than a
	 *             shift, so the gallery can offer a continuous size control.
	 * @param cancelled polled while rendering; when it turns true the render
	 *                  gives up and returns false, leaving @a out unspecified.
	 *
	 * @return false if the render was cancelled or could not run, true otherwise.
	 */
	static bool render(Squey::PVView const& view,
	                   PVCol x_col,
	                   PVCol y_col,
	                   int size,
	                   PVScatterThumbnailImages& out,
	                   std::atomic<bool> const& cancelled);

	/**
	 * Row count above which correlation() samples.
	 *
	 * Much lower than row_sampling_threshold, because ranking needs every pair
	 * scored, not just the visible ones: the whole ranking costs
	 * pairs x threshold. A correlation coefficient converges long before this
	 * many rows anyway -- the extra digits it would gain never change an
	 * ordering the user reads as a rough "most structured first".
	 */
	constexpr static PVRow correlation_sampling_threshold = 1000 * 1000;

	/**
	 * Total scaled values a whole ranking pass may read, across every pair.
	 *
	 * Pairs grow as the square of the axis count -- 150 columns already make
	 * 11175 of them -- and a fixed per-pair sample makes the pass cost
	 * pairs x rows. Measured on 150 columns x 300k rows: 290 ms without this
	 * budget, 102 ms with. So it is a guard rail, not a rescue: the case it
	 * actually matters for is the same file an order of magnitude longer, where
	 * the fixed sample would run into tens of seconds behind a gallery that
	 * only says "ranking pairs".
	 *
	 * What the budget buys is a pass whose cost stops following the row count,
	 * for a coarser sample. That price is small: the ranking only has to get
	 * the order roughly right, and a correlation is already stable to a
	 * hundredth on ten thousand rows.
	 */
	constexpr static size_t correlation_sample_budget = 200 * 1000 * 1000;

	//! Never sample below this, however many pairs share the budget.
	constexpr static PVRow correlation_min_sample = 10 * 1000;

	/**
	 * Rows one pair may visit when @a pair_count of them share the budget.
	 */
	static PVRow correlation_sample_size(PVRow row_count, size_t pair_count);

	/**
	 * The per-column half of a correlation, so that ranking N axes walks each
	 * column once instead of once per pair it takes part in.
	 */
	struct ColumnMoments {
		double sum = 0.;
		double sum_sq = 0.;
		size_t count = 0;
	};

	/**
	 * @param sample_size rows to visit; must be the value correlation() is
	 *        given for the same pass, or the two halves would not describe the
	 *        same rows.
	 */
	static ColumnMoments column_moments(Squey::PVView const& view, PVCol col, PVRow sample_size);

	/**
	 * Correlation between two scaled columns, in [0, 1], used to rank pairs.
	 *
	 * This is Pearson's r over the scaled values, of which the absolute value
	 * is returned: the gallery ranks by how structured a pair looks, and an
	 * anti-correlation is just as worth looking at as a correlation. Scaled
	 * values are the rank-uniformised form of the data, so this behaves like a
	 * rank correlation -- monotonic but non-linear relations still score high,
	 * which is what one wants when ranking scatter plots by interest.
	 *
	 * @param x_moments, y_moments must come from column_moments() on the same
	 *        two columns, which is where the per-column sums are computed.
	 */
	static double correlation(Squey::PVView const& view,
	                          PVCol x_col,
	                          PVCol y_col,
	                          ColumnMoments const& x_moments,
	                          ColumnMoments const& y_moments,
	                          PVRow sample_size);
};

} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVSCATTERTHUMBNAIL_H
