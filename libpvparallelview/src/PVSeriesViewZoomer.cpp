#include <pvparallelview/PVSeriesViewZoomer.h>

#include <pvparallelview/PVSeriesView.h>
#include <inendi/PVRangeSubSampler.h>

#include <QMouseEvent>
#include <QToolTip>
#include <QPainter>

namespace PVParallelView
{

struct PVSeriesViewZoomerRectangleFragment : public QWidget {
	PVSeriesViewZoomerRectangleFragment(QWidget* parent = nullptr) : QWidget(parent) {}

	void paintEvent(QPaintEvent*) override
	{
		QPainter painter(this);
		painter.fillRect(rect(), QColor(255, 0, 0, 255));
	}
};

PVSeriesViewZoomer::PVSeriesViewZoomer(PVSeriesView* child,
                                       Inendi::PVRangeSubSampler& sampler,
                                       QWidget* parent)
    : QWidget(parent), m_seriesView(child), m_rss(sampler)
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
	if (event->button() != Qt::RightButton) {
		return;
	}
	m_selecting = true;
	QToolTip::showText(event->globalPos(), tr("Zoom"), this);
	m_zoomRect.moveTo(event->pos());
}

void PVSeriesViewZoomer::mouseReleaseEvent(QMouseEvent* event)
{
	if (not m_selecting or event->button() != Qt::RightButton) {
		return;
	}
	m_selecting = false;
	QToolTip::hideText();
	for (auto& fragment : m_fragments) {
		fragment->hide();
	}
	zoomIn(m_zoomRect.normalized());
}

void PVSeriesViewZoomer::mouseMoveEvent(QMouseEvent* event)
{
	if (not m_selecting) {
		return;
	}
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
	updateZoom(m_zoomStack[++m_currentZoomIndex]);
}

void PVSeriesViewZoomer::zoomIn(QPoint center)
{
	qDebug() << "zoomIn" << center;
	if (m_currentZoomIndex + 1 < m_zoomStack.size()) {
		updateZoom(m_zoomStack[++m_currentZoomIndex]);
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
	updateZoom(m_zoomStack[++m_currentZoomIndex]);
}

void PVSeriesViewZoomer::zoomOut()
{
	qDebug() << "zoomOut";
	if (m_currentZoomIndex == 0) {
		return;
	}
	updateZoom(m_zoomStack[--m_currentZoomIndex]);
}

void PVSeriesViewZoomer::resetZoom()
{
	m_zoomStack.clear();
	m_zoomStack.push_back(Zoom{0., 1., 0., 1.});
	m_currentZoomIndex = 0;
	updateZoom(m_zoomStack.back());
}

void PVSeriesViewZoomer::updateZoom(Zoom const& zoom)
{
	m_rss.subsample(std::clamp(zoom.minX, m_zoomStack.front().minX, m_zoomStack.front().maxX),
	                std::clamp(zoom.maxX, m_zoomStack.front().minX, m_zoomStack.front().maxX),
	                std::clamp(zoom.minY, m_zoomStack.front().minY, m_zoomStack.front().maxY),
	                std::clamp(zoom.maxY, m_zoomStack.front().minY, m_zoomStack.front().maxY));
	m_seriesView->update();
	qDebug() << "Zoom stack (current:" << m_currentZoomIndex << "):";
	for (auto& zoom : m_zoomStack) {
		qDebug() << "\t" << zoom.minX << zoom.maxX << zoom.minY << zoom.maxY;
	}
}
}
