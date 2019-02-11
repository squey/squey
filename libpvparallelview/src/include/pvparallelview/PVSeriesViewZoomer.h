/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2018
 */

#ifndef _PVPARALLELVIEW_PVSERIESVIEWZOOMER_H_
#define _PVPARALLELVIEW_PVSERIESVIEWZOOMER_H_

#include <QWidget>
#include <QBasicTimer>

namespace Inendi
{
class PVRangeSubSampler;
}

namespace PVParallelView
{

class PVSeriesView;

class PVViewZoomer : public QWidget
{
	Q_OBJECT
  public:
	struct Zoom {
		double minX;
		double maxX;
		double minY;
		double maxY;

		double width() const { return maxX - minX; }
		double height() const { return maxY - minY; }
	};

	PVViewZoomer(QWidget* parent = nullptr);

	void zoomIn(QRect zoomInRect);
	void zoomIn(QPoint center, bool rectangular, double zoomFactor);
	void zoomOut();
	void zoomOut(QPoint center);
	void resetZoom();

	void moveZoomBy(QPoint offset);

	Zoom currentZoom() const { return m_zoomStack[m_currentZoomIndex]; }
	QRect normalizedZoomRect(QRect zoomRect, bool rectangular) const;

	static void clampZoom(Zoom& zoom);

  Q_SIGNALS:
	void zoomUpdated(Zoom zoom);

  protected:
	virtual void updateZoom(Zoom) {}

  private:
	void updateZoom();

  private:
	std::vector<Zoom> m_zoomStack;
	size_t m_currentZoomIndex = 0;
};

class PVSeriesViewZoomer : public PVViewZoomer
{
  public:
	PVSeriesViewZoomer(PVSeriesView* child,
	                   Inendi::PVRangeSubSampler& sampler,
	                   QWidget* parent = nullptr);
	virtual ~PVSeriesViewZoomer() = default;

	QColor getZoomRectColor() const;
	void setZoomRectColor(QColor const& color);

  protected:
	void mousePressEvent(QMouseEvent*) override;
	void mouseReleaseEvent(QMouseEvent*) override;
	void mouseMoveEvent(QMouseEvent*) override;
	void leaveEvent(QEvent* event) override;
	void keyPressEvent(QKeyEvent* event) override;
	void keyReleaseEvent(QKeyEvent* event) override;

	void wheelEvent(QWheelEvent*) override;

	void resizeEvent(QResizeEvent*) override;
	void timerEvent(QTimerEvent* event) override;

	void updateZoom(Zoom zoom) override;

  private:
	void updateZoomGeometry(bool rectangular);

  private:
	PVSeriesView* m_seriesView;
	Inendi::PVRangeSubSampler& m_rss;

	QBasicTimer m_resizingTimer;

	bool m_selecting = false;
	QRect m_zoomRect;
	std::array<QWidget*, 4> m_fragments{nullptr};
	std::array<QWidget*, 4> m_crossHairsFragments{nullptr};

	bool m_moving = false;
	QPoint m_moveStart;

	QTimer* m_animationTimer;

	const double m_centeredZoomFactor = 0.8;
};
}

#endif // _PVPARALLELVIEW_PVSERIESVIEWZOOMER_H_
