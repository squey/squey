/**
 * \file PVAxisSlider.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvparallelview/PVAxisSlider.h>

// pour PVParallelView::AxisWidth
#include <pvparallelview/common.h>

#include <QPainter>
#include <QGraphicsScene>

#include <iostream>

#define SLIDER_HALF_WIDTH 8
#define SLIDER_WIDTH (2 * SLIDER_HALF_WIDTH + PVParallelView::AxisWidth)

/*****************************************************************************
 * PVParallelView::PVAxisSlider::PVAxisSlider
 *****************************************************************************/

PVParallelView::PVAxisSlider::PVAxisSlider(int omin, int omax, int o) :
	_offset_min(omin), _offset_max(omax)
{
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

void PVParallelView::PVAxisSlider::hoverEnterEvent(QGraphicsSceneHoverEvent * event)
{
	std::cout << "PVAxisSlider::hoverEnterEvent " << this << std::endl;
}

/*****************************************************************************
 * PVParallelView::PVAxisSlider::hoverMoveEvent
 *****************************************************************************/

void PVParallelView::PVAxisSlider::hoverMoveEvent(QGraphicsSceneHoverEvent * event)
{
	std::cout << "PVAxisSlider::hoverMoveEvent " << this << std::endl;
}

/*****************************************************************************
 * PVParallelView::PVAxisSlider::hoverLeaveEvent
 *****************************************************************************/

void PVParallelView::PVAxisSlider::hoverLeaveEvent(QGraphicsSceneHoverEvent * event)
{
	std::cout << "PVAxisSlider::hoverLeaveEvent " << this << std::endl;
}

