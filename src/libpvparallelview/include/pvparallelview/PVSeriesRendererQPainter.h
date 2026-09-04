/* * MIT License
 *
 * © ESI Group, 2015
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

#ifndef _PVSERIESRENDERERQPAINTER_H_
#define _PVSERIESRENDERERQPAINTER_H_

#include <pvparallelview/PVSeriesAbstractRenderer.h>

#include <QPainter>

namespace PVParallelView
{

/**
 * Reference renderer: one QPainter polyline per serie. Portable and easy to reason
 * about, but single threaded and paying the generic stroker on every segment, so it
 * only stays interactive for a handful of series -- see PVSeriesRendererRaster.
 */
class PVSeriesRendererQPainter : public PVSeriesAbstractRenderer
{
	using PVRSS = Squey::PVRangeSubSampler;

  public:
	PVSeriesRendererQPainter(Squey::PVRangeSubSampler const& rss)
	    : PVSeriesAbstractRenderer(rss)
	{
	}

	static constexpr bool capability() { return true; }
	static PVSeriesView::DrawMode capability(PVSeriesView::DrawMode mode)
	{
		if (mode == PVSeriesView::DrawMode::Lines || mode == PVSeriesView::DrawMode::Points ||
		    mode == PVSeriesView::DrawMode::LinesAlways) {
			return mode;
		}
		return PVSeriesView::DrawMode::Lines;
	}

	void set_background_color(QColor const& bgcol) override { _background_color = bgcol; }
	void set_draw_mode(PVSeriesView::DrawMode mode) override { _draw_mode = capability(mode); }

	QImage grab() override
	{
		if (_image.size() != _size) {
			_image = QImage(_size, QImage::Format_RGB32);
		}
		if (_image.isNull()) {
			return _image;
		}

		QPainter painter(&_image);
		painter.fillRect(_image.rect(), _background_color);

		if (not _rss.valid()) {
			return _image;
		}

		std::vector<QPoint> points;
		auto draw_lines = [&painter, &points]() {
			if (points.size() > 1) {
				painter.drawPolyline(points.data(), points.size());
			} else if (points.size() == 1) {
				painter.drawPoint(points.front());
			}
		};
		for (auto& serie_draw : _series_draw_order) {
			painter.setPen(serie_draw.color);
			auto& serie_data = _rss.sampled_timeserie(serie_draw.dataIndex);
			for (size_t j = 0; j < serie_data.size();) {
				while (j < serie_data.size()) {
					int vertex = serie_data[j];
					QPoint point(j, (height() - 1) -
					                    vertex * (height() - 1) / PVRSS::display_type_max_val);
					if (PVRSS::display_match(vertex, PVRSS::overflow_value)) {
						point = QPoint(j, -1);
					} else if (PVRSS::display_match(vertex, PVRSS::underflow_value)) {
						point = QPoint(j, height());
					} else if (PVRSS::display_match(vertex, PVRSS::no_value)) {
						while (++j < serie_data.size() and
						       PVRSS::display_match(serie_data[j], PVRSS::no_value)) {
						}
						break;
					}
					points.push_back(point);
					++j;
				}
				if (_draw_mode == PVSeriesView::DrawMode::Lines) {
					draw_lines();
					points.clear();
				}
			}
			if (_draw_mode == PVSeriesView::DrawMode::Points) {
				painter.drawPoints(points.data(), points.size());
				points.clear();
			} else if (_draw_mode == PVSeriesView::DrawMode::LinesAlways) {
				if (not points.empty()) {
					points.insert(points.begin(), QPoint(0, points[0].y()));
					points.insert(points.end(), QPoint(width() - 1, points.back().y()));
				}
				draw_lines();
				points.clear();
			}
		}
		painter.end();

		return _image;
	}

  private:
	PVSeriesView::DrawMode _draw_mode = PVSeriesView::DrawMode::Lines;
	QColor _background_color = Qt::black;
	QImage _image;
};

} // namespace PVParallelView

#endif // _PVSERIESRENDERERQPAINTER_H_
