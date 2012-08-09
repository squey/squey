/**
 * \file PVAxisSlider.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVAlgorithms.h>
#include <pvparallelview/PVAxisSlider.h>

#include <QPainter>
#include <QGraphicsScene>
#include <QGraphicsSceneMouseEvent>

#include <iostream>

/*****************************************************************************
 * PVParallelView::PVAxisSlider::PVAxisSlider
 *****************************************************************************/

PVParallelView::PVAxisSlider::PVAxisSlider(int omin, int omax, int o) :
	_offset_min(omin), _offset_max(omax), _moving(false)
{
	setAcceptHoverEvents(true); // This is needed to enable hover events

	if(o < omin) {
		_offset = omin;
	} else if (o > omax) {
		_offset = omax;
	} else {
		_offset = o;
	}
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

QRectF PVParallelView::PVAxisSlider::boundingRect () const
{
	return QRectF(- SLIDER_HALF_WIDTH, _offset - 1,
	              SLIDER_WIDTH, PVParallelView::AxisWidth);
}

/*****************************************************************************
 * PVParallelView::PVAxisSlider::paint
 *****************************************************************************/

void PVParallelView::PVAxisSlider::paint(QPainter *painter,
                                         const QStyleOptionGraphicsItem */*option*/,
                                         QWidget */*widget*/)
{
	painter->fillRect(- SLIDER_HALF_WIDTH, _offset - 1,
	                  SLIDER_WIDTH,  PVParallelView::AxisWidth,
	                  Qt::SolidPattern);

}

/*****************************************************************************
 * PVParallelView::PVAxisSlider::hoverenterEvent
 *****************************************************************************/

void PVParallelView::PVAxisSlider::hoverEnterEvent(QGraphicsSceneHoverEvent* event)
{
	PVLOG_INFO("PVAxisSlider::hoverEnterEvent\n");
}

/*****************************************************************************
 * PVParallelView::PVAxisSlider::hoverMoveEvent
 *****************************************************************************/

void PVParallelView::PVAxisSlider::hoverMoveEvent(QGraphicsSceneHoverEvent* event)
{
	PVLOG_INFO("PVAxisSlider::hoverMoveEvent\n");
}

/*****************************************************************************
 * PVParallelView::PVAxisSlider::hoverLeaveEvent
 *****************************************************************************/

void PVParallelView::PVAxisSlider::hoverLeaveEvent(QGraphicsSceneHoverEvent* event)
{
	PVLOG_INFO("PVAxisSlider::hoverLeaveEvent\n");
}

/*****************************************************************************
 * PVParallelView::PVAxisSlider::mousePressEvent
 *****************************************************************************/

void PVParallelView::PVAxisSlider::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
	_moving = true;
	//event->accept();
}

/*****************************************************************************
 * PVParallelView::PVAxisSlider::mouseReleaseEvent
 *****************************************************************************/

void PVParallelView::PVAxisSlider::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
	_moving = false;
	emit slider_moved();
	//event->accept();
}

/*****************************************************************************
 * PVParallelView::PVAxisSlider::mouseMoveEvent
 *****************************************************************************/

void PVParallelView::PVAxisSlider::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
	if (event->buttons() == Qt::LeftButton) {
		_offset = PVCore::clamp(event->pos().y(), (qreal) 0, (qreal) PVParallelView::ImageHeight-1);
		group()->update();
	}
	event->accept();
}
