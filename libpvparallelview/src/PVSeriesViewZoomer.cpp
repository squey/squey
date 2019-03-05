#include <pvparallelview/PVSeriesViewZoomer.h>

#include <pvparallelview/PVSeriesView.h>
#include <inendi/PVRangeSubSampler.h>

#include <QMouseEvent>
#include <QToolTip>
#include <QPainter>

#include <QTimer>
#include <QDebug>

namespace PVParallelView
{

PVViewZoomer::PVViewZoomer(QWidget* parent) : QWidget(parent)
{
	m_zoomStack.push_back(Zoom{0., 1., 0., 1.});
}

void PVViewZoomer::zoomIn(QRect zoomInRect)
{
	qDebug() << "zoomIn" << zoomInRect;
	m_zoomStack.resize(m_currentZoomIndex + 1);
	m_zoomStack.push_back(rectToZoom(zoomInRect));
	++m_currentZoomIndex;
	updateZoom();
}

void PVViewZoomer::zoomIn(QPoint center, bool rectangular, zoom_f zoomFactor)
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

	struct PointF {
		zoom_f _x;
		zoom_f _y;
		zoom_f x() const { return _x; }
		zoom_f y() const { return _y; }
		zoom_f& rx() { return _x; }
		zoom_f& ry() { return _y; }
	};

	PointF centerRatio{zoom_f(center.x()) / zoom_f(size().width()),
	                   1. - zoom_f(center.y()) / zoom_f(size().height())};
	PointF fixedCenter{oldZoom.minX + centerRatio.x() * oldZoom.width(),
	                   oldZoom.minY + centerRatio.y() * oldZoom.height()};
	PointF targetTopLeft{fixedCenter.x() - centerRatio.x() * targetZoom.width(),
	                     fixedCenter.y() - centerRatio.y() * targetZoom.height()};

	targetTopLeft.rx() = std::clamp(targetTopLeft.x(), zoom_f(0), zoom_f(1) - targetZoom.width());
	targetTopLeft.ry() = std::clamp(targetTopLeft.y(), zoom_f(0), zoom_f(1) - targetZoom.height());

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

void PVViewZoomer::resetAndZoomIn(Zoom zoom)
{
	m_zoomStack.clear();
	m_zoomStack.push_back(Zoom{0., 1., 0., 1.});
	m_zoomStack.push_back(zoom);
	m_currentZoomIndex = 1;
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
		zoom_f offsetX = offset.x() * zoom.width() / zoom_f(size().width());
		offsetX = std::clamp(offsetX, -(1. - zoom.maxX), zoom.minX);
		zoom.minX -= offsetX;
		zoom.maxX -= offsetX;
	}
	{
		zoom_f offsetY = offset.y() * zoom.height() / zoom_f(size().height());
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

auto PVViewZoomer::rectToZoom(QRect const& rect) const -> Zoom
{
	Zoom const& currentZoom = m_zoomStack[m_currentZoomIndex];
	return Zoom{
	    currentZoom.minX + currentZoom.width() * (rect.x() / zoom_f(size().width())),
	    currentZoom.minX +
	        currentZoom.width() * ((rect.x() + rect.width()) / zoom_f(size().width())),
	    currentZoom.minY +
	        currentZoom.height() * (1. - (rect.y() + rect.height()) / zoom_f(size().height())),
	    currentZoom.minY + currentZoom.height() * (1. - rect.y() / zoom_f(size().height()))};
}

void PVViewZoomer::clampZoom(Zoom& zoom)
{
	zoom.minX = std::clamp(zoom.minX, zoom_f(0), zoom_f(1));
	zoom.maxX = std::clamp(zoom.maxX, zoom_f(0), zoom_f(1));
	zoom.minY = std::clamp(zoom.minY, zoom_f(0), zoom_f(1));
	zoom.maxY = std::clamp(zoom.maxY, zoom_f(0), zoom_f(1));
}

void PVViewZoomer::updateZoom()
{
	updateZoom(currentZoom());
	zoomUpdated(currentZoom());

	qDebug() << "Zoom stack (current:" << m_currentZoomIndex << "):";
	for (auto& zoom : m_zoomStack) {
		qDebug() << "\t" << std::to_string(zoom.minX).c_str() << std::to_string(zoom.maxX).c_str()
		         << std::to_string(zoom.minY).c_str() << std::to_string(zoom.maxY).c_str();
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
	for (auto& fragment : m_zoomFragments) {
		fragment = new PVSeriesViewZoomerRectangleFragment(child);
		fragment->hide();
	}
	for (auto& fragment : m_selectionFragments) {
		fragment = new PVSeriesViewZoomerRectangleFragment(child, QColor(20, 255, 50, 255));
		fragment->hide();
	}
	for (auto& chronotip : m_chronotips) {
		chronotip = new QLabel(child);
		chronotip->setSizePolicy(QSizePolicy::Policy::Expanding, QSizePolicy::Policy::Minimum);
		chronotip->setAutoFillBackground(true);
		chronotip->hide();
	}
	m_chronotips[0]->setAlignment(Qt::AlignRight);
	m_chronotips[1]->setAlignment(Qt::AlignLeft);
	setMouseTracking(true);
	setCursor(Qt::BlankCursor);
}

void PVSeriesViewZoomer::mousePressEvent(QMouseEvent* event)
{
	if (event->button() == Qt::LeftButton) {
		if (event->modifiers() & Qt::ShiftModifier) {
			m_selecting = true;
			m_selectionRect.moveTo(event->pos());
		} else {
			m_zooming = true;
			// QToolTip::showText(event->globalPos(), tr("Zoom"), this);
			m_zoomRect.moveTo(event->pos());
		}
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
	if (m_zooming && event->button() == Qt::LeftButton) {
		m_zooming = false;
		// QToolTip::hideText();
		hideFragments(m_chronotips);
		hideFragments(m_zoomFragments);
		if (event->pos() == m_zoomRect.topLeft()) {
			zoomOut(event->pos());
		} else {
			zoomIn(normalizedZoomRect(m_zoomRect, event->modifiers() & Qt::ControlModifier));
		}
	} else if (m_selecting && event->button() == Qt::LeftButton) {
		m_selecting = false;
		hideFragments(m_selectionFragments);
		selectionCommit(rectToZoom(normalizedZoomRect(m_selectionRect, false)));
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
	if (m_zooming) {
		m_zoomRect.setBottomRight(QPoint(std::clamp(event->pos().x(), 0, size().width() - 1),
		                                 std::clamp(event->pos().y(), 0, size().height() - 1)));
		updateZoomGeometry(event->modifiers() & Qt::ControlModifier);
		showFragments(m_zoomFragments);
		updateChronotips(m_zoomRect);
	}
	if (m_selecting) {
		m_selectionRect.setBottomRight(
		    QPoint(std::clamp(event->pos().x(), 0, size().width() - 1),
		           std::clamp(event->pos().y(), 0, size().height() - 1)));
		updateSelectionGeometry();
		showFragments(m_selectionFragments);
		updateChronotips(m_selectionRect);
	}
	if (m_moving) {
		moveZoomBy(event->pos() - m_moveStart);
		m_moveStart = event->pos();
		updateChronotips(event->pos());
	}
	if (not m_zooming and not m_selecting) {
		updateCrossHairsGeometry(event->pos());
		showFragments(m_crossHairsFragments);
		updateChronotips(event->pos());
	} else {
		hideFragments(m_crossHairsFragments);
	}
	showFragments(m_chronotips);
}

void PVSeriesViewZoomer::keyPressEvent(QKeyEvent* event)
{
	if (event->key() == Qt::Key_Control) {
		if (m_zooming) {
			updateZoomGeometry(true);
		}
	}
}

void PVSeriesViewZoomer::keyReleaseEvent(QKeyEvent* event)
{
	if (event->key() == Qt::Key_Control) {
		if (m_zooming) {
			updateZoomGeometry(false);
		}
	}
}

void PVSeriesViewZoomer::leaveEvent(QEvent*)
{
	hideFragments(m_crossHairsFragments);
	hideFragments(m_chronotips);
}

void PVSeriesViewZoomer::wheelEvent(QWheelEvent* event)
{
	if (event->angleDelta().y() > 0) {
		zoomIn(event->pos(), event->modifiers() & Qt::ControlModifier, m_centeredZoomFactor);
	} else if (event->angleDelta().y() < 0) {
		zoomOut(event->pos());
	}
}

void PVSeriesViewZoomer::resizeEvent(QResizeEvent*)
{
	m_resizingTimer.stop();
	m_resizingTimer.start(200, this);
	m_seriesView->resize(size());
}

void PVSeriesViewZoomer::timerEvent(QTimerEvent* event)
{
	if (event->timerId() == m_resizingTimer.timerId()) {
		m_rss.set_sampling_count(size().width());
		updateZoom(currentZoom());
		m_resizingTimer.stop();
	}
}

void PVSeriesViewZoomer::updateZoomGeometry(bool rectangular)
{
	auto zr = normalizedZoomRect(m_zoomRect, rectangular);
	m_zoomFragments[0]->setGeometry(zr.x(), zr.y(), zr.width(), 1);
	m_zoomFragments[1]->setGeometry(zr.x(), zr.y(), 1, zr.height());
	m_zoomFragments[2]->setGeometry(zr.x(), zr.y() + zr.height(), zr.width(), 1);
	m_zoomFragments[3]->setGeometry(zr.x() + zr.width(), zr.y(), 1, zr.height());
}

void PVSeriesViewZoomer::updateSelectionGeometry()
{
	auto zr = normalizedZoomRect(m_selectionRect, false);
	m_selectionFragments[0]->setGeometry(zr.x(), zr.y(), 1, zr.height());
	m_selectionFragments[1]->setGeometry(zr.x() + zr.width(), zr.y(), 1, zr.height());
}

void PVSeriesViewZoomer::updateCrossHairsGeometry(QPoint pos)
{
	m_crossHairsFragments[0]->setGeometry(0, pos.y(), pos.x() - 10, 1);
	m_crossHairsFragments[1]->setGeometry(pos.x(), 0, 1, pos.y() - 10);
	m_crossHairsFragments[2]->setGeometry(pos.x() + 10, pos.y(), width() - pos.x() - 10, 1);
	m_crossHairsFragments[3]->setGeometry(pos.x(), pos.y() + 10, 1, height() - pos.y() - 10);
}

void PVSeriesViewZoomer::updateChronotipGeometry(size_t chrono_index, QPoint pos)
{
	bool chronotips_top = (pos.y() < height() / 2)
	                          ? pos.y() > 3 * m_chronotips[0]->height()
	                          : pos.y() > height() - 3 * m_chronotips[0]->height();
	bool force_chronotips_right = pos.x() < m_chronotips[0]->width();
	bool force_chronotips_left = pos.x() > width() - m_chronotips[1]->width();
	if (chrono_index == 0) {
		int chronotips_y_0 = chronotips_top ? 0 : force_chronotips_left or force_chronotips_right
		                                              ? height() - m_chronotips[0]->height() -
		                                                    m_chronotips[1]->height()
		                                              : height() - m_chronotips[0]->height();
		m_chronotips[0]->move(force_chronotips_right ? pos.x() + 1
		                                             : pos.x() - m_chronotips[0]->width(),
		                      chronotips_y_0);
	} else if (chrono_index == 1) {
		int chronotips_y_1 =
		    chronotips_top
		        ? force_chronotips_left or force_chronotips_right ? m_chronotips[0]->height() : 0
		        : height() - m_chronotips[1]->height();
		m_chronotips[1]->move(force_chronotips_left ? pos.x() - m_chronotips[1]->width()
		                                            : pos.x() + 1,
		                      chronotips_y_1);
	} else if (chrono_index == 2) {
		bool percenttips_left = pos.x() > 2 * m_chronotips[2]->width();
		m_chronotips[2]->move(percenttips_left ? 0 : width() - m_chronotips[2]->width(),
		                      pos.y() + 1);
	} else if (chrono_index == 3) {
		bool percenttips_left = pos.x() > 2 * m_chronotips[3]->width();
		m_chronotips[3]->move(percenttips_left ? 0 : width() - m_chronotips[3]->width(),
		                      pos.y() - m_chronotips[3]->height());
	}
}

template <class T>
void PVSeriesViewZoomer::showFragments(T const& fragments) const
{
	for (auto& fragment : fragments) {
		fragment->show();
	}
}

template <class T>
void PVSeriesViewZoomer::hideFragments(T const& fragments) const
{
	for (auto& fragment : fragments) {
		fragment->hide();
	}
}

void PVSeriesViewZoomer::updateZoom(Zoom zoom)
{
	m_rss.subsample(zoom.minX, zoom.maxX, zoom.minY, zoom.maxY);
}

void PVSeriesViewZoomer::updateChronotips(QPoint point)
{
	updateChronotips(QRect(point.x(), point.y(), 1, 1));
}

void PVSeriesViewZoomer::updateChronotips(QRect rect)
{
	rect = normalizedZoomRect(rect, true);
	Zoom coveredZoom = rectToZoom(rect);
	pvcop::db::array subrange = m_rss.ratio_to_minmax(coveredZoom.minX, coveredZoom.maxX);
	m_chronotips[0]->setText((subrange.at(0) + ">").c_str());
	m_chronotips[1]->setText(("<" + subrange.at(1)).c_str());
	m_chronotips[2]->setText((std::to_string(coveredZoom.minY) + "%").c_str());
	m_chronotips[3]->setText((std::to_string(coveredZoom.maxY) + "%").c_str());
	for (auto* chronotip : m_chronotips) {
		chronotip->adjustSize();
	}
	updateChronotipGeometry(0, rect.topLeft());
	updateChronotipGeometry(1, rect.bottomRight());
	updateChronotipGeometry(2, rect.bottomRight());
	updateChronotipGeometry(3, rect.topLeft());
}

QColor PVSeriesViewZoomer::getZoomRectColor() const
{
	return static_cast<PVSeriesViewZoomerRectangleFragment*>(m_zoomFragments[0])->color;
}

void PVSeriesViewZoomer::setZoomRectColor(QColor const& color)
{
	for (auto& fragment : m_zoomFragments) {
		static_cast<PVSeriesViewZoomerRectangleFragment*>(fragment)->color = color;
	}
}
}
