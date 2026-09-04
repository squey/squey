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

#include <pvparallelview/PVSeriesRendererRaster.h>

#include <pvparallelview/PVSeriesSampleRows.h>

#include <algorithm>

namespace PVParallelView
{

PVSeriesRendererRaster::PVSeriesRendererRaster(Squey::PVRangeSubSampler const& rss)
    : PVSeriesAbstractRenderer(rss)
{
}

bool PVSeriesRendererRaster::capability()
{
	return true;
}

PVSeriesView::DrawMode PVSeriesRendererRaster::capability(PVSeriesView::DrawMode mode)
{
	if (mode == PVSeriesView::DrawMode::Lines or mode == PVSeriesView::DrawMode::Points or
	    mode == PVSeriesView::DrawMode::LinesAlways) {
		return mode;
	}
	return PVSeriesView::DrawMode::Lines;
}

void PVSeriesRendererRaster::set_background_color(QColor const& bgcol)
{
	_background_color = bgcol;
}

void PVSeriesRendererRaster::set_draw_mode(PVSeriesView::DrawMode mode)
{
	_draw_mode = capability(mode);
}

void PVSeriesRendererRaster::rasterise_tile(
    QRgb* pixels, int stride, int x_begin, int x_end, int samples_count)
{
	const int h = height();
	const int series_count = int(_series_draw_order.size());
	const bool points_only = _draw_mode == PVSeriesView::DrawMode::Points;

	const QRgb background = _background_color.rgb();
	for (int y = 0; y < h; ++y) {
		QRgb* row = pixels + size_t(y) * stride;
		std::fill(row + x_begin, row + x_end, background);
	}

	// The sampling count trails the widget width by one resize: the columns beyond it
	// have no sample yet and stay on the background.
	const int x_last = std::min(x_end, samples_count);

	for (int s = 0; s < series_count; ++s) {
		const int16_t* rows = _rows.data() + size_t(s) * samples_count;
		const QRgb color = _colors[s];
		for (int x = x_begin; x < x_last; ++x) {
			const int from = rows[x];
			if (from == PVSeriesSampleRows::no_row) {
				continue;
			}
			// The sample of the next column closes the segment; the last one, and any
			// sample the next column does not join, is a lone pixel.
			const int to = (points_only or x + 1 >= samples_count or rows[x + 1] == PVSeriesSampleRows::no_row)
			                   ? from
			                   : rows[x + 1];
			const int top = std::max(std::min(from, to), 0);
			const int bottom = std::min(std::max(from, to), h - 1);
			for (int y = top; y <= bottom; ++y) {
				pixels[size_t(y) * stride + x] = color;
			}
		}
	}
}

QImage PVSeriesRendererRaster::grab()
{
	if (_image.size() != _size) {
		_image = QImage(_size, QImage::Format_RGB32);
	}
	if (_image.isNull()) {
		return _image;
	}

	const int w = width();
	const int samples_count =
	    _rss.valid() and not _series_draw_order.empty()
	        ? int(std::min<size_t>(_rss.sampled_timeserie(_series_draw_order.front().dataIndex).size(),
	                               size_t(w)))
	        : 0;

	if (samples_count <= 0) {
		_image.fill(_background_color);
		return _image;
	}

	PVSeriesSampleRows::project(_rss, _series_draw_order, _draw_mode, samples_count, height(),
	                            _rows);
	_colors.resize(_series_draw_order.size());
	for (size_t s = 0; s < _series_draw_order.size(); ++s) {
		_colors[s] = _series_draw_order[s].color.rgb();
	}

	const int tile_count = (w + tile_width - 1) / tile_width;
	// Below a few tens of thousands of spans the thread pool costs more than the work it
	// spreads, and the series view is redrawn on every mouse move.
	const bool worth_threading = size_t(samples_count) * _series_draw_order.size() > 20000;

	// Taken before the tiles fan out: bits() detaches, which no two threads may race on.
	QRgb* const pixels = reinterpret_cast<QRgb*>(_image.bits());
	const int stride = _image.bytesPerLine() / int(sizeof(QRgb));

#pragma omp parallel for schedule(static) if (worth_threading)
	for (int tile = 0; tile < tile_count; ++tile) {
		rasterise_tile(pixels, stride, tile * tile_width,
		               std::min((tile + 1) * tile_width, w), samples_count);
	}

	return _image;
}

} // namespace PVParallelView
