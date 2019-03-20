/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2019
 */

#ifndef _PVSERIESRENDERERQPAINTER_H_
#define _PVSERIESRENDERERQPAINTER_H_

#include <pvparallelview/PVSeriesAbstractRenderer.h>

#include <QPainter>

namespace PVParallelView
{

class PVSeriesRendererQPainter : public PVSeriesAbstractRenderer, public QWidget
{
	using PVRSS = Inendi::PVRangeSubSampler;

  public:
	PVSeriesRendererQPainter(Inendi::PVRangeSubSampler const& rss, QWidget* parent = nullptr)
	    : PVSeriesAbstractRenderer(rss), QWidget(parent)
	{
		setAutoFillBackground(true);
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

	void set_background_color(QColor const& bgcol) override { setPalette(QPalette(bgcol)); }
	void set_draw_mode(PVSeriesView::DrawMode mode) override { _draw_mode = capability(mode); }

	void resize(QSize const& size) override { return QWidget::resize(size); }
	QPixmap grab() override { return QWidget::grab(); }

  protected:
	void paintEvent(QPaintEvent*) override
	{
		if (not _rss.valid()) {
			return;
		}
		QPainter painter(this);
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
	}

  private:
	PVSeriesView::DrawMode _draw_mode = PVSeriesView::DrawMode::Lines;
};

} // namespace PVParallelView

#endif // _PVSERIESRENDERERQPAINTER_H_