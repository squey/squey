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

	void setBackgroundColor(QColor const& bgcol) override { setPalette(QPalette(bgcol)); }
	void setDrawMode(PVSeriesView::DrawMode mode) override { m_drawMode = capability(mode); }

	void resize(QSize const& size) override { return QWidget::resize(size); }
	QPixmap grab() override { return QWidget::grab(); }

  protected:
	void paintEvent(QPaintEvent*) override
	{
		if (not m_rss.valid()) {
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
		for (auto& serieDraw : m_seriesDrawOrder) {
			painter.setPen(serieDraw.color);
			auto& serieData = m_rss.sampled_timeserie(serieDraw.dataIndex);
			for (size_t j = 0; j < serieData.size();) {
				while (j < serieData.size()) {
					int vertex = serieData[j];
					QPoint point(j, (height() - 1) - vertex * (height() - 1) / ((1 << 14) - 1));
					if (vertex & (1 << 15)) {     // if out of range
						if (vertex & (1 << 14)) { // if overflow
							point = QPoint(j, -1);
						} else { // else underflow
							point = QPoint(j, height());
						}
					} else if (vertex & (1 << 14)) { // else if no value
						while (++j < serieData.size() and
						       (serieData[j] & (1 << 14) and not(serieData[j] & (1 << 15))))
							;
						break;
					}
					points.push_back(point);
					++j;
				}
				if (m_drawMode == PVSeriesView::DrawMode::Lines) {
					draw_lines();
					points.clear();
				}
			}
			if (m_drawMode == PVSeriesView::DrawMode::Points) {
				painter.drawPoints(points.data(), points.size());
				points.clear();
			} else if (m_drawMode == PVSeriesView::DrawMode::LinesAlways) {
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
	PVSeriesView::DrawMode m_drawMode = PVSeriesView::DrawMode::Lines;
};

} // namespace PVParallelView

#endif // _PVSERIESRENDERERQPAINTER_H_