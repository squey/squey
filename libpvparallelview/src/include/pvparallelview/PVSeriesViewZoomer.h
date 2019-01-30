/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2018
 */

#ifndef _PVPARALLELVIEW_PVSERIESVIEWZOOMER_H_
#define _PVPARALLELVIEW_PVSERIESVIEWZOOMER_H_

#include <QWidget>

namespace Inendi
{
class PVRangeSubSampler;
}

namespace PVParallelView
{

class PVSeriesView;

class PVSeriesViewZoomer : public QWidget
{
	struct Zoom {
		double minX;
		double maxX;
		double minY;
		double maxY;

		double width() const { return maxX - minX; }
		double height() const { return maxY - minY; }
	};

  public:
	PVSeriesViewZoomer(PVSeriesView* child,
	                   Inendi::PVRangeSubSampler& sampler,
	                   QWidget* parent = nullptr);
	virtual ~PVSeriesViewZoomer() = default;

	void zoomIn(QRect zoomInRect);
	void zoomIn(QPoint center);
	void zoomOut();
	void zoomOut(QPoint center);
	void resetZoom();

	void moveZoomBy(QPoint offset);

	QColor getZoomRectColor() const;
	void setZoomRectColor(QColor const& color);

  protected:
	void mousePressEvent(QMouseEvent*) override;
	void mouseReleaseEvent(QMouseEvent*) override;
	void mouseMoveEvent(QMouseEvent*) override;
	void leaveEvent(QEvent* event) override;

	void wheelEvent(QWheelEvent*) override;

	void resizeEvent(QResizeEvent*) override;

  private:
	void updateZoom();
	void clampZoom(Zoom& zoom) const;

  private:
	PVSeriesView* m_seriesView;
	Inendi::PVRangeSubSampler& m_rss;

	bool m_selecting = false;
	QRect m_zoomRect;
	std::array<QWidget*, 4> m_fragments{nullptr};
	std::array<QWidget*, 4> m_crossHairsFragments{nullptr};

	bool m_moving = false;
	QPoint m_moveStart;

	QTimer* m_animationTimer;

	std::vector<Zoom> m_zoomStack;
	size_t m_currentZoomIndex = 0;

	const double m_centeredZoomFactor = 0.8;
};
}

#endif // _PVPARALLELVIEW_PVSERIESVIEWZOOMER_H_
