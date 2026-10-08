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

#ifndef _PVSERIESSAMPLEROWS_H_
#define _PVSERIESSAMPLEROWS_H_

#include <pvparallelview/PVSeriesView.h>

#include <cstdint>
#include <limits>
#include <vector>

namespace PVParallelView
{

/**
 * Turning the sampled values into pixel rows is the one step every renderer needs and
 * none of them wants to redo: it resolves the three out-of-band values the subsampler
 * uses, and it settles what LinesAlways means, so that what follows is a plain row per
 * sample. Both the CPU rasteriser and the GPU backend start from here.
 */
namespace PVSeriesSampleRows
{

// Row of a sample that has no value at all: neither drawn nor joined to.
static constexpr int16_t no_row = std::numeric_limits<int16_t>::min();

/**
 * Fills @a rows with series_count * samples_count rows, one contiguous run per entry of
 * @a draw_order. Rows may fall outside [0, height): the subsampler reports values below
 * and above the zoom range, and a segment reaching one of them still has to cross the
 * visible rows, exactly as it did when QPainter clipped it.
 */
void project(Squey::PVRangeSubSampler const& rss,
             std::vector<PVSeriesView::SerieDrawInfo> const& draw_order,
             PVSeriesView::DrawMode draw_mode,
             int samples_count,
             int height,
             std::vector<int16_t>& rows);

} // namespace PVSeriesSampleRows

} // namespace PVParallelView

#endif // _PVSERIESSAMPLEROWS_H_
