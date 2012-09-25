
#include <pvkernel/core/PVLogger.h>

#include <pvparallelview/PVAbstractAxisSlider.h>

#include <QPainter>
#include <QGraphicsScene>
#include <QGraphicsSceneMouseEvent>

/*****************************************************************************
 * PVParallelView::PVAbstractAxisSlider::PVAbstractAxisSlider
 *****************************************************************************/

PVParallelView::PVAbstractAxisSlider::PVAbstractAxisSlider(int omin, int omax, int o,
                                                           PVAxisSliderOrientation orientation) :
	_offset_min(omin), _offset_max(omax), _orientation(orientation),
	_moving(false)
{
	setAcceptHoverEvents(true); // This is needed to enable hover events

	setFlag(QGraphicsItem::ItemIgnoresTransformations, true);

	set_value(o);
}

/*****************************************************************************
 * PVParallelView::PVAbstractAxisSlider::~PVAbstractAxisSlider
 *****************************************************************************/

PVParallelView::PVAbstractAxisSlider::~PVAbstractAxisSlider()
{
	QGraphicsScene *s = scene();

	if (s != 0) {
		s->removeItem(this);
	}
}

/*****************************************************************************
 * PVParallelView::PVAbstractAxisSlider::hoverenterEvent
 *****************************************************************************/

void PVParallelView::PVAbstractAxisSlider::hoverEnterEvent(QGraphicsSceneHoverEvent* /*event*/)
{
	PVLOG_INFO("PVAbstractAxisSlider::hoverEnterEvent\n");
}

/*****************************************************************************
 * PVParallelView::PVAbstractAxisSlider::hoverMoveEvent
 *****************************************************************************/

void PVParallelView::PVAbstractAxisSlider::hoverMoveEvent(QGraphicsSceneHoverEvent* /*event*/)
{
	PVLOG_INFO("PVAbstractAxisSlider::hoverMoveEvent\n");
}

/*****************************************************************************
 * PVParallelView::PVAbstractAxisSlider::hoverLeaveEvent
 *****************************************************************************/

void PVParallelView::PVAbstractAxisSlider::hoverLeaveEvent(QGraphicsSceneHoverEvent* /*event*/)
{
	PVLOG_INFO("PVAbstractAxisSlider::hoverLeaveEvent\n");
}

/*****************************************************************************
 * PVParallelView::PVAbstractAxisSlider::mousePressEvent
 *****************************************************************************/

void PVParallelView::PVAbstractAxisSlider::mousePressEvent(QGraphicsSceneMouseEvent* /*event*/)
{
	_moving = true;
	//event->accept();
}

/*****************************************************************************
 * PVParallelView::PVAbstractAxisSlider::mouseReleaseEvent
 *****************************************************************************/

void PVParallelView::PVAbstractAxisSlider::mouseReleaseEvent(QGraphicsSceneMouseEvent* /*event*/)
{
	emit slider_moved();
	_moving = false;
	//event->accept();
}

/*****************************************************************************
 * PVParallelView::PVAbstractAxisSlider::mouseMoveEvent
 *****************************************************************************/

void PVParallelView::PVAbstractAxisSlider::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
	if (event->buttons() == Qt::LeftButton) {
		// +0.5 to have a rounded value
		set_value(event->scenePos().y() + 0.5);

		group()->update();
	}
	event->accept();
}
