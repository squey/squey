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
	static PVSeriesView::DrawMode capability(PVSeriesView::DrawMode)
	{
		return PVSeriesView::DrawMode::Lines;
	}

	void setBackgroundColor(QColor const& bgcol) override { setPalette(QPalette(bgcol)); }
	void setDrawMode(PVSeriesView::DrawMode) override {}

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
		for (auto& serieDraw : m_seriesDrawOrder) {
			painter.setPen(serieDraw.color);
			auto& serieData = m_rss.averaged_timeserie(serieDraw.dataIndex);
			for (size_t j = 0; j < serieData.size();) {
				points.clear();
				while (j < serieData.size()) {
					int vertex = serieData[j];
					if (vertex & (1 << 15)) {     // if out of range
						if (vertex & (1 << 14)) { // if overflow
							vertex = (1 << 15);
						} else { // else underflow
							vertex = -(1 << 15);
						}
					} else if (vertex & (1 << 14)) { // else if no value
						while (++j < serieData.size() && (serieData[j] & (1 << 14)))
							;
						break;
					}
					points.push_back(QPoint(j, height() - vertex * height() / (1 << 14)));
					++j;
				}
				if (points.size() > 1) {
					painter.drawPolyline(points.data(), points.size());
				} else if (points.size() == 1) {
					painter.drawPoint(points.front());
				}
			}
		}
		painter.end();
	}
};

} // namespace PVParallelView

#endif // _PVSERIESRENDERERQPAINTER_H_