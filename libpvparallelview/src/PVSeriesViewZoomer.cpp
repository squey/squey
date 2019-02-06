#include <pvparallelview/PVSeriesViewZoomer.h>

#include <pvparallelview/PVSeriesView.h>
#include <inendi/PVRangeSubSampler.h>

#include <QMouseEvent>
#include <QToolTip>
#include <QPainter>

#include <QTimer>

namespace PVParallelView
{

PVViewZoomer::PVViewZoomer(QWidget* parent) : QWidget(parent)
{
}

void PVViewZoomer::zoomIn(QRect zoomInRect)
{
	qDebug() << "zoomIn" << zoomInRect;
	m_zoomStack.resize(m_currentZoomIndex + 1);
	Zoom const& currentZoom = m_zoomStack[m_currentZoomIndex];
	m_zoomStack.push_back(Zoom{
	    currentZoom.minX + currentZoom.width() * (zoomInRect.x() / double(size().width())),
	    currentZoom.minX +
	        currentZoom.width() * ((zoomInRect.x() + zoomInRect.width()) / double(size().width())),
	    currentZoom.minY +
	        currentZoom.height() *
	            (1. - (zoomInRect.y() + zoomInRect.height()) / double(size().height())),
	    currentZoom.minY + currentZoom.height() * (1. - zoomInRect.y() / double(size().height()))});
	++m_currentZoomIndex;
	updateZoom();
}

void PVViewZoomer::zoomIn(QPoint center, bool rectangular, double zoomFactor)
{
	qDebug() << "zoomIn" << center;
	// if (m_currentZoomIndex + 1 < m_zoomStack.size()) {
	// 	++m_currentZoomIndex;
	// 	updateZoom();
	// 	return;
	// }
	assert(zoomFactor < 1 && zoomFactor > 0);
	zoomIn(QRect(center.x() - center.x() * zoomFactor,
	             center.y() - center.y() * (rectangular ? zoomFactor : 1),
	             zoomFactor * size().width(), (rectangular ? zoomFactor : 1) * size().height()));
}

void PVViewZoomer::zoomOut()
{
	qDebug() << "zoomOut";
	if (m_currentZoomIndex == 0) {
		return;
	}
	--m_currentZoomIndex;
	updateZoom();
}

void PVViewZoomer::zoomOut(QPoint center)
{
	qDebug() << "zoomOut" << center;
	if (m_currentZoomIndex == 0) {
		return;
	}
	Zoom oldZoom = m_zoomStack[m_currentZoomIndex];
	--m_currentZoomIndex;
	Zoom targetZoom = m_zoomStack[m_currentZoomIndex];

	QPointF centerRatio{double(center.x()) / double(size().width()),
	                    1. - double(center.y()) / double(size().height())};
	QPointF fixedCenter{oldZoom.minX + centerRatio.x() * oldZoom.width(),
	                    oldZoom.minY + centerRatio.y() * oldZoom.height()};
	QPointF targetTopLeft{fixedCenter.x() - centerRatio.x() * targetZoom.width(),
	                      fixedCenter.y() - centerRatio.y() * targetZoom.height()};

	targetTopLeft.rx() = std::clamp(targetTopLeft.x(), 0., 1. - targetZoom.width());
	targetTopLeft.ry() = std::clamp(targetTopLeft.y(), 0., 1. - targetZoom.height());

	m_zoomStack[m_currentZoomIndex] =
	    Zoom{targetTopLeft.x(), targetTopLeft.x() + targetZoom.width(), targetTopLeft.y(),
	         targetTopLeft.y() + targetZoom.height()};

	updateZoom();
}

void PVViewZoomer::resetZoom()
{
	m_zoomStack.clear();
	m_zoomStack.push_back(Zoom{0., 1., 0., 1.});
	m_currentZoomIndex = 0;
	updateZoom();
}

void PVViewZoomer::moveZoomBy(QPoint offset)
{
	if (m_currentZoomIndex == 0) {
		return;
	}
	m_zoomStack.resize(m_currentZoomIndex + 1);
	Zoom& zoom = m_zoomStack.back();
	{
		double offsetX = offset.x() * zoom.width() / double(size().width());
		offsetX = std::clamp(offsetX, -(1. - zoom.maxX), zoom.minX);
		zoom.minX -= offsetX;
		zoom.maxX -= offsetX;
	}
	{
		double offsetY = offset.y() * zoom.height() / double(size().height());
		offsetY = std::clamp(offsetY, -zoom.minY, 1. - zoom.maxY);
		zoom.minY += offsetY;
		zoom.maxY += offsetY;
	}
	updateZoom();
}

QRect PVViewZoomer::normalizedZoomRect(QRect zoomRect, bool rectangular) const
{
	auto zr = zoomRect.normalized();
	if (not rectangular) {
		zr.setY(0);
		zr.setHeight(size().height());
	}
	return zr;
}

void PVViewZoomer::clampZoom(Zoom& zoom)
{
	zoom.minX = std::clamp(zoom.minX, 0., 1.);
	zoom.maxX = std::clamp(zoom.maxX, 0., 1.);
	zoom.minY = std::clamp(zoom.minY, 0., 1.);
	zoom.maxY = std::clamp(zoom.maxY, 0., 1.);
}

void PVViewZoomer::updateZoom()
{
	updateZoom(currentZoom());
	zoomUpdated(currentZoom());

	qDebug() << "Zoom stack (current:" << m_currentZoomIndex << "):";
	for (auto& zoom : m_zoomStack) {
		qDebug() << "\t" << zoom.minX << zoom.maxX << zoom.minY << zoom.maxY;
	}
}

struct PVSeriesViewZoomerRectangleFragment : public QWidget {
	PVSeriesViewZoomerRectangleFragment(QWidget* parent = nullptr,
	                                    QColor color = QColor(255, 0, 0, 255))
	    : QWidget(parent), color(color)
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
    : PVViewZoomer(parent), m_seriesView(child), m_rss(sampler), m_animationTimer(new QTimer(this))
{
	child->setParent(this);
	child->setAttribute(Qt::WA_TransparentForMouseEvents);
	for (auto& fragment : m_crossHairsFragments) {
		fragment = new PVSeriesViewZoomerRectangleFragment(child, QColor(255, 100, 50, 255));
		fragment->hide();
	}
	for (auto& fragment : m_fragments) {
		fragment = new PVSeriesViewZoomerRectangleFragment(child);
		fragment->hide();
	}
	resetZoom();
	setMouseTracking(true);
	setFocusPolicy(Qt::ClickFocus);
}

void PVSeriesViewZoomer::mousePressEvent(QMouseEvent* event)
{
	if (event->button() == Qt::LeftButton) {
		m_selecting = true;
		// QToolTip::showText(event->globalPos(), tr("Zoom"), this);
		m_zoomRect.moveTo(event->pos());
	} else if (event->button() == Qt::RightButton) {
		m_moving = true;
		// QToolTip::showText(event->globalPos(), tr("Moving"), this);
		m_moveStart = event->pos();
	} else if (event->button() == Qt::MidButton) {
		using namespace std::chrono;
		m_animationTimer->callOnTimeout([this] { moveZoomBy({-1, 0}); });
		m_animationTimer->start(36ms);
	}
}

void PVSeriesViewZoomer::mouseReleaseEvent(QMouseEvent* event)
{
	if (m_selecting && event->button() == Qt::LeftButton) {
		m_selecting = false;
		// QToolTip::hideText();
		for (auto& fragment : m_fragments) {
			fragment->hide();
		}
		if (event->pos() == m_zoomRect.topLeft()) {
			zoomOut(event->pos());
		} else {
			zoomIn(normalizedZoomRect(m_zoomRect, event->modifiers() & Qt::ControlModifier));
		}
	} else if (m_moving && event->button() == Qt::RightButton) {
		m_moving = false;
		// QToolTip::hideText();
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
		updateZoomGeometry(event->modifiers() & Qt::ControlModifier);
		for (auto& fragment : m_fragments) {
			fragment->show();
		}
	}
	if (m_moving) {
		moveZoomBy(event->pos() - m_moveStart);
		m_moveStart = event->pos();
	}
	if (not m_selecting and not m_moving) {
		m_crossHairsFragments[0]->setGeometry(0, event->pos().y(), event->pos().x() - 10, 1);
		m_crossHairsFragments[1]->setGeometry(event->pos().x(), 0, 1, event->pos().y() - 10);
		m_crossHairsFragments[2]->setGeometry(event->pos().x() + 10, event->pos().y(),
		                                      width() - event->pos().x() - 10, 1);
		m_crossHairsFragments[3]->setGeometry(event->pos().x(), event->pos().y() + 10, 1,
		                                      height() - event->pos().y() - 10);
		for (auto& fragment : m_crossHairsFragments) {
			fragment->show();
		}
	} else {
		for (auto& fragment : m_crossHairsFragments) {
			fragment->hide();
		}
	}
}

void PVSeriesViewZoomer::keyPressEvent(QKeyEvent* event)
{
	if (event->key() == Qt::Key_Control) {
		if (m_selecting) {
			updateZoomGeometry(true);
		}
	}
}

void PVSeriesViewZoomer::keyReleaseEvent(QKeyEvent* event)
{
	if (event->key() == Qt::Key_Control) {
		if (m_selecting) {
			updateZoomGeometry(false);
		}
	}
}

void PVSeriesViewZoomer::leaveEvent(QEvent*)
{
	for (auto& fragment : m_crossHairsFragments) {
		fragment->hide();
	}
}

void PVSeriesViewZoomer::wheelEvent(QWheelEvent* event)
{
	if (event->angleDelta().y() > 0) {
		zoomIn(event->pos(), event->modifiers() & Qt::ControlModifier, m_centeredZoomFactor);
	} else if (event->angleDelta().y() < 0) {
		zoomOut(event->pos());
	}
}

void PVSeriesViewZoomer::resizeEvent(QResizeEvent* event)
{
	m_seriesView->resize(event->size());
}

void PVSeriesViewZoomer::updateZoomGeometry(bool rectangular)
{
	auto zr = normalizedZoomRect(m_zoomRect, rectangular);
	m_fragments[0]->setGeometry(zr.x(), zr.y(), zr.width(), 1);
	m_fragments[1]->setGeometry(zr.x(), zr.y(), 1, zr.height());
	m_fragments[2]->setGeometry(zr.x(), zr.y() + zr.height(), zr.width(), 1);
	m_fragments[3]->setGeometry(zr.x() + zr.width(), zr.y(), 1, zr.height());
}

void PVSeriesViewZoomer::updateZoom(Zoom zoom)
{
	m_rss.subsample(zoom.minX, zoom.maxX, zoom.minY, zoom.maxY);
	m_seriesView->onResampled();
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
