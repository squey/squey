#include <pvparallelview/PVSeriesViewZoomer.h>

#include <pvparallelview/PVSeriesView.h>
#include <inendi/PVRangeSubSampler.h>

#include <QMouseEvent>
#include <QToolTip>
#include <QPainter>

#include <QTimer>

namespace PVParallelView
{

struct PVSeriesViewZoomerRectangleFragment : public QWidget {
	PVSeriesViewZoomerRectangleFragment(QWidget* parent = nullptr)
	    : QWidget(parent), color(255, 0, 0, 255)
	{
	}

	void paintEvent(QPaintEvent*) override
	{
		QPainter painter(this);
		painter.fillRect(rect(), color);
	}

	QColor color;
};

PVSeriesViewZoomer::PVSeriesViewZoomer(PVSeriesView* child,
                                       Inendi::PVRangeSubSampler& sampler,
                                       QWidget* parent)
    : QWidget(parent), m_seriesView(child), m_rss(sampler), m_animationTimer(new QTimer(this))
{
	child->setParent(this);
	for (auto& fragment : m_fragments) {
		fragment = new PVSeriesViewZoomerRectangleFragment(child);
		fragment->hide();
	}
	resetZoom();
}

void PVSeriesViewZoomer::mousePressEvent(QMouseEvent* event)
{
	if (event->button() == Qt::RightButton) {
		m_selecting = true;
		QToolTip::showText(event->globalPos(), tr("Zoom"), this);
		m_zoomRect.moveTo(event->pos());
	} else if (event->button() == Qt::LeftButton) {
		m_moving = true;
		QToolTip::showText(event->globalPos(), tr("Moving"), this);
		m_moveStart = event->pos();
	} else if (event->button() == Qt::MidButton) {
		using namespace std::chrono;
		m_animationTimer->callOnTimeout([this] { moveZoomBy({-1, 0}); });
		m_animationTimer->start(36ms);
	}
}

void PVSeriesViewZoomer::mouseReleaseEvent(QMouseEvent* event)
{
	if (m_selecting && event->button() == Qt::RightButton) {
		m_selecting = false;
		QToolTip::hideText();
		for (auto& fragment : m_fragments) {
			fragment->hide();
		}
		if (event->pos() == m_zoomRect.topLeft()) {
			zoomOut();
		} else {
			zoomIn(m_zoomRect.normalized());
		}
	} else if (m_moving && event->button() == Qt::LeftButton) {
		m_moving = false;
		QToolTip::hideText();
		moveZoomBy(event->pos() - m_moveStart);
	} else if (event->button() == Qt::MidButton) {
		m_animationTimer->stop();
	}
}

void PVSeriesViewZoomer::mouseMoveEvent(QMouseEvent* event)
{
	if (m_selecting) {
		m_zoomRect.setBottomRight(QPoint(std::clamp(event->pos().x(), 0, size().width() - 1),
		                                 std::clamp(event->pos().y(), 0, size().height() - 1)));
		auto zr = m_zoomRect.normalized();
		m_fragments[0]->setGeometry(zr.x(), zr.y(), zr.width(), 1);
		m_fragments[1]->setGeometry(zr.x(), zr.y(), 1, zr.height());
		m_fragments[2]->setGeometry(zr.x(), zr.y() + zr.height(), zr.width(), 1);
		m_fragments[3]->setGeometry(zr.x() + zr.width(), zr.y(), 1, zr.height());
		for (auto& fragment : m_fragments) {
			fragment->show();
		}
	}
	if (m_moving) {
		moveZoomBy(event->pos() - m_moveStart);
		m_moveStart = event->pos();
	}
}

void PVSeriesViewZoomer::wheelEvent(QWheelEvent* event)
{
	if (event->angleDelta().y() > 0) {
		zoomIn(event->pos());
	} else if (event->angleDelta().y() < 0) {
		zoomOut();
	}
}

void PVSeriesViewZoomer::resizeEvent(QResizeEvent* event)
{
	m_seriesView->resize(event->size());
}

void PVSeriesViewZoomer::zoomIn(QRect zoomInRect)
{
	qDebug() << "zoomIn" << zoomInRect;
	m_zoomStack.resize(m_currentZoomIndex + 1);
	Zoom const& currentZoom = m_zoomStack[m_currentZoomIndex];
	m_zoomStack.push_back(
	    Zoom{currentZoom.minX +
	             (currentZoom.maxX - currentZoom.minX) * (zoomInRect.x() / double(size().width())),
	         currentZoom.minX +
	             (currentZoom.maxX - currentZoom.minX) *
	                 ((zoomInRect.x() + zoomInRect.width()) / double(size().width())),
	         currentZoom.minY +
	             (currentZoom.maxY - currentZoom.minY) *
	                 (1. - (zoomInRect.y() + zoomInRect.height()) / double(size().height())),
	         currentZoom.minY +
	             (currentZoom.maxY - currentZoom.minY) *
	                 (1. - zoomInRect.y() / double(size().height()))});
	++m_currentZoomIndex;
	updateZoom();
}

void PVSeriesViewZoomer::zoomIn(QPoint center)
{
	qDebug() << "zoomIn" << center;
	if (m_currentZoomIndex + 1 < m_zoomStack.size()) {
		++m_currentZoomIndex;
		updateZoom();
		return;
	}
	center = {std::clamp<int>(center.x(), m_centeredZoomRadius * size().width(),
	                          (1. - m_centeredZoomRadius) * size().width()),
	          std::clamp<int>(center.y(), m_centeredZoomRadius * size().height(),
	                          (1. - m_centeredZoomRadius) * size().height())};
	Zoom const& currentZoom = m_zoomStack[m_currentZoomIndex];
	m_zoomStack.push_back(
	    Zoom{currentZoom.minX +
	             (currentZoom.maxX - currentZoom.minX) *
	                 (center.x() / double(size().width()) - m_centeredZoomRadius),
	         currentZoom.minX +
	             (currentZoom.maxX - currentZoom.minX) *
	                 (center.x() / double(size().width()) + m_centeredZoomRadius),
	         currentZoom.minY +
	             (currentZoom.maxY - currentZoom.minY) *
	                 (1. - center.y() / double(size().height()) - m_centeredZoomRadius),
	         currentZoom.minY +
	             (currentZoom.maxY - currentZoom.minY) *
	                 (1. - center.y() / double(size().height()) + m_centeredZoomRadius)});
	++m_currentZoomIndex;
	updateZoom();
}

void PVSeriesViewZoomer::zoomOut()
{
	qDebug() << "zoomOut";
	if (m_currentZoomIndex == 0) {
		return;
	}
	--m_currentZoomIndex;
	updateZoom();
}

void PVSeriesViewZoomer::resetZoom()
{
	m_zoomStack.clear();
	m_zoomStack.push_back(Zoom{0., 1., 0., 1.});
	m_currentZoomIndex = 0;
	updateZoom();
}

void PVSeriesViewZoomer::moveZoomBy(QPoint offset)
{
	if (m_currentZoomIndex == 0) {
		return;
	}
	m_zoomStack.resize(m_currentZoomIndex + 1);
	Zoom& zoom = m_zoomStack.back();
	zoom.minX -= offset.x() * (zoom.maxX - zoom.minX) / double(size().width());
	zoom.maxX -= offset.x() * (zoom.maxX - zoom.minX) / double(size().width());
	zoom.minY += offset.y() * (zoom.maxY - zoom.minY) / double(size().height());
	zoom.maxY += offset.y() * (zoom.maxY - zoom.minY) / double(size().height());
	updateZoom();
}

void PVSeriesViewZoomer::updateZoom()
{
	Zoom& zoom = m_zoomStack[m_currentZoomIndex];
	clampZoom(zoom);
	m_rss.subsample(zoom.minX, zoom.maxX, zoom.minY, zoom.maxY);
	m_seriesView->onResampled();
	qDebug() << "Zoom stack (current:" << m_currentZoomIndex << "):";
	for (auto& zoom : m_zoomStack) {
		qDebug() << "\t" << zoom.minX << zoom.maxX << zoom.minY << zoom.maxY;
	}
}

void PVSeriesViewZoomer::clampZoom(Zoom& zoom) const
{
	zoom.minX = std::clamp(zoom.minX, 0., 1.);
	zoom.maxX = std::clamp(zoom.maxX, 0., 1.);
	zoom.minY = std::clamp(zoom.minY, 0., 1.);
	zoom.maxY = std::clamp(zoom.maxY, 0., 1.);
}

QColor PVSeriesViewZoomer::getZoomRectColor() const
{
	return static_cast<PVSeriesViewZoomerRectangleFragment*>(m_fragments[0])->color;
}

void PVSeriesViewZoomer::setZoomRectColor(QColor const& color)
{
	for (auto& fragment : m_fragments) {
		static_cast<PVSeriesViewZoomerRectangleFragment*>(fragment)->color = color;
	}
}
}
