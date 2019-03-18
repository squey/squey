/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2018
 */

#ifndef _PVPARALLELVIEW_PVSERIESVIEWZOOMER_H_
#define _PVPARALLELVIEW_PVSERIESVIEWZOOMER_H_

#include <QWidget>
#include <QLabel>
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
	using zoom_f = long double;
	struct Zoom {
		zoom_f minX;
		zoom_f maxX;
		zoom_f minY;
		zoom_f maxY;

		zoom_f width() const { return maxX - minX; }
		zoom_f height() const { return maxY - minY; }
	};

	PVViewZoomer(QWidget* parent = nullptr);

	void zoomIn(QRect zoomInRect);
	void zoomIn(QPoint center, bool rectangular, zoom_f zoomFactor);
	void zoomOut();
	void zoomOut(QPoint center);
	void zoomOut(QPoint center, bool rectangular, zoom_f zoomFactor);
	void resetZoom();
	void resetAndZoomIn(Zoom zoom);

	void moveZoomBy(QPoint offset);

	QRect normalizedZoomRect(QRect zoomRect, bool rectangular) const;
	Zoom rectToZoom(QRect const& rect) const;
	Zoom currentZoom() const { return m_zoomStack[m_currentZoomIndex]; }

	static void clampZoom(Zoom& zoom);

  Q_SIGNALS:
	void zoomUpdated(Zoom zoom);

  protected:
	virtual void updateZoom(Zoom) {}

  private:
	void zoomOut(QPoint center, Zoom oldZoom, Zoom targetZoom);
	void updateZoom();

  private:
	std::vector<Zoom> m_zoomStack;
	size_t m_currentZoomIndex = 0;
};

class PVSeriesViewZoomer : public PVViewZoomer
{
	Q_OBJECT
  public:
	PVSeriesViewZoomer(PVSeriesView* child,
	                   Inendi::PVRangeSubSampler& sampler,
	                   QWidget* parent = nullptr);
	virtual ~PVSeriesViewZoomer() = default;

	enum class SelectorMode { CrossHairs = 0, Zooming = 1, Selecting = 2, Hunting = 3 };

	SelectorMode currentSelectorMode() const { return m_selectorMode; }
	void changeSelectorMode(SelectorMode const mode);

	QColor getSelectorColor(SelectorMode mode) const { return m_selectorColors[size_t(mode)]; }
	void setSelectorColor(SelectorMode mode, QColor color)
	{
		m_selectorColors[size_t(mode)] = color;
	}

	int getCrossHairsRadius() const { return m_crossHairsRadius; }
	void setCrossHairsRadius(int radius) { m_crossHairsRadius = radius; }

  Q_SIGNALS:
	void selectorModeChanged(SelectorMode previousMode, SelectorMode currentMode);
	void selectionCommit(Zoom selection);
	void cursorMoved(QRect region);
	void huntCommit(QRect region, bool addition);

  protected:
	void mousePressEvent(QMouseEvent*) override;
	void mouseReleaseEvent(QMouseEvent*) override;
	void mouseMoveEvent(QMouseEvent*) override;
	void enterEvent(QEvent* event) override;
	void leaveEvent(QEvent* event) override;
	void keyPressEvent(QKeyEvent* event) override;
	void keyReleaseEvent(QKeyEvent* event) override;

	void wheelEvent(QWheelEvent*) override;

	void resizeEvent(QResizeEvent*) override;
	void timerEvent(QTimerEvent* event) override;

	void updateZoom(Zoom zoom) override;
	void updateChronotips(QPoint point);
	void updateChronotips(QRect rect);

  private:
	void updateSelectorAndChronotips();
	void updateSelectorGeometry(bool rectangular);
	void updateCrossHairsGeometry(QPoint pos);
	void updateChronotipGeometry(size_t chrono_index, QPoint pos);
	template <class T>
	void showFragments(T const& fragments) const;
	template <class T>
	void hideFragments(T const& fragments) const;
	QRect crossHairsRect(QPoint pos) const;

  private:
	PVSeriesView* m_seriesView;
	Inendi::PVRangeSubSampler& m_rss;

	QBasicTimer m_resizingTimer;

	SelectorMode m_selectorMode = SelectorMode::CrossHairs;
	QRect m_selectorRect;
	std::array<QWidget*, 4> m_selectorFragments{nullptr};
	std::array<QColor, 4> m_selectorColors{
	    QColor(255, 100, 50, 255), // CrossHairs
	    QColor(255, 0, 0, 255),    // Zooming
	    QColor(20, 255, 50, 255),  // Selecting
	    QColor(20, 20, 255, 255)   // Hunting
	};
	int m_crossHairsRadius = 10;
	std::array<QLabel*, 4> m_chronotips{nullptr};

	bool m_control_modifier = false;
	bool m_left_button_down = false;

	bool m_moving = false;
	QPoint m_moveStart;

	QTimer* m_animationTimer;

	const zoom_f m_centeredZoomFactor = 0.8;
};
}

#endif // _PVPARALLELVIEW_PVSERIESVIEWZOOMER_H_
