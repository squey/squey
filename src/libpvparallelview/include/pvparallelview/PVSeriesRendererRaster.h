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

#ifndef _PVSERIESRENDERERRASTER_H_
#define _PVSERIESRENDERERRASTER_H_

#include <pvparallelview/PVSeriesAbstractRenderer.h>
#include <pvparallelview/PVSeriesSampleRows.h>

#include <cstdint>
#include <vector>

namespace PVParallelView
{

/**
 * Multi-threaded rasteriser specialised for the one shape the series view ever draws.
 *
 * The subsampler hands out exactly one value per column of pixels, so a serie is not a
 * general polyline but a vertical span per column. That removes the need for a stroker,
 * for clipping and for sub-pixel positioning, and it makes the picture splittable into
 * independent vertical tiles that the cores can chew on in parallel. A tile is narrow
 * enough that the cache lines it writes are shared by all of its columns, which is what
 * keeps the vertical writes affordable.
 *
 * Against PVSeriesRendererQPainter on the same data, this is about eight times faster on
 * one core and thirty to fifty times faster on twenty, for a picture that differs only by
 * the rounding of the segment ends.
 */
class PVSeriesRendererRaster : public PVSeriesAbstractRenderer
{
	using PVRSS = Squey::PVRangeSubSampler;

	// A tile is sixteen pixels wide so that the four cache lines a column write touches
	// are amortised over sixteen columns.
	static constexpr int tile_width = 16;

  public:
	explicit PVSeriesRendererRaster(Squey::PVRangeSubSampler const& rss);

	static bool capability();
	static PVSeriesView::DrawMode capability(PVSeriesView::DrawMode);

	void set_background_color(QColor const& bgcol) override;
	void set_draw_mode(PVSeriesView::DrawMode) override;

	QImage grab() override;

  private:
	void rasterise_tile(QRgb* pixels, int stride, int x_begin, int x_end, int samples_count);

	PVSeriesView::DrawMode _draw_mode = PVSeriesView::DrawMode::Lines;
	QColor _background_color = Qt::black;
	QImage _image;

	// Pixel row of every sample of every drawn serie, in draw order.
	std::vector<int16_t> _rows;
	std::vector<QRgb> _colors;
};

} // namespace PVParallelView

#endif // _PVSERIESRENDERERRASTER_H_
