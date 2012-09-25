/**
 * \file PVAxisSlider.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvparallelview/PVAxisSlider.h>

#include <QPainter>
#include <QGraphicsScene>
#include <QGraphicsSceneMouseEvent>

#include <iostream>

/*****************************************************************************
 * PVParallelView::PVAxisSlider::PVAxisSlider
 *****************************************************************************/

PVParallelView::PVAxisSlider::PVAxisSlider(int omin, int omax, int o,
                                           PVAxisSliderType type) :
	_offset_min(omin), _offset_max(omax), _type(type), _moving(false)
{
	setAcceptHoverEvents(true); // This is needed to enable hover events

	setFlag(QGraphicsItem::ItemIgnoresTransformations, true);

	set_value(o);
}

/*****************************************************************************
 * PVParallelView::PVAxisSlider::~PVAxisSlider
 *****************************************************************************/

PVParallelView::PVAxisSlider::~PVAxisSlider()
{
	QGraphicsScene *s = scene();

	if (s != 0) {
		s->removeItem(this);
	}
}


bool PVParallelView::PVAxisSlider::is_moving() const
{
	return _moving;
}

/*****************************************************************************
 * PVParallelView::PVAxisSlider::boundingRect
 *****************************************************************************/

QRectF PVParallelView::PVAxisSlider::boundingRect() const
{
	if (_type == Min) {
		return QRectF(QPointF(- SLIDER_HALF_WIDTH, 0),
		              QPointF(SLIDER_HALF_WIDTH, -4));
	} else {
		return QRectF(QPointF(- SLIDER_HALF_WIDTH, 0),
		              QPointF(SLIDER_HALF_WIDTH, 4));
	}
}

/*****************************************************************************
 * PVParallelView::PVAxisSlider::paint
 *****************************************************************************/

void PVParallelView::PVAxisSlider::paint(QPainter *painter,
                                         const QStyleOptionGraphicsItem */*option*/,
                                         QWidget */*widget*/)
{
	painter->fillRect(boundingRect(),
	                  Qt::black);

	QPen old_pen = painter->pen();
	painter->setPen(Qt::white);
	painter->drawRect(boundingRect());
	painter->setPen(old_pen);
}

/*****************************************************************************
 * PVParallelView::PVAxisSlider::hoverenterEvent
 *****************************************************************************/

void PVParallelView::PVAxisSlider::hoverEnterEvent(QGraphicsSceneHoverEvent* /*event*/)
{
	PVLOG_INFO("PVAxisSlider::hoverEnterEvent\n");
}

/*****************************************************************************
 * PVParallelView::PVAxisSlider::hoverMoveEvent
 *****************************************************************************/

void PVParallelView::PVAxisSlider::hoverMoveEvent(QGraphicsSceneHoverEvent* /*event*/)
{
	PVLOG_INFO("PVAxisSlider::hoverMoveEvent\n");
}

/*****************************************************************************
 * PVParallelView::PVAxisSlider::hoverLeaveEvent
 *****************************************************************************/

void PVParallelView::PVAxisSlider::hoverLeaveEvent(QGraphicsSceneHoverEvent* /*event*/)
{
	PVLOG_INFO("PVAxisSlider::hoverLeaveEvent\n");
}

/*****************************************************************************
 * PVParallelView::PVAxisSlider::mousePressEvent
 *****************************************************************************/

void PVParallelView::PVAxisSlider::mousePressEvent(QGraphicsSceneMouseEvent* /*event*/)
{
	_moving = true;
	//event->accept();
}

/*****************************************************************************
 * PVParallelView::PVAxisSlider::mouseReleaseEvent
 *****************************************************************************/

void PVParallelView::PVAxisSlider::mouseReleaseEvent(QGraphicsSceneMouseEvent* /*event*/)
{
	emit slider_moved();
	_moving = false;
	//event->accept();
}

/*****************************************************************************
 * PVParallelView::PVAxisSlider::mouseMoveEvent
 *****************************************************************************/

void PVParallelView::PVAxisSlider::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
	if (event->buttons() == Qt::LeftButton) {
		// +0.5 to have a rounded value
		set_value(event->scenePos().y() + 0.5);

		group()->update();
	}
	event->accept();
}
