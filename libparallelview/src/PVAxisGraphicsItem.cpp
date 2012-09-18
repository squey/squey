/**
 * \file PVAxisGraphicsItem.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <iostream>

#include <picviz/PVAxis.h>
#include <pvparallelview/PVAxisGraphicsItem.h>

#include <QPainter>
#include <QGraphicsScene>

// Used to draw the axis out of the image zone
#define PVAW_CST 8

/*****************************************************************************
 * PVParallelView::PVAxisGraphicsItem::PVAxisGraphicsItem
 *****************************************************************************/

PVParallelView::PVAxisGraphicsItem::PVAxisGraphicsItem(PVParallelView::PVSlidersManager_p sm_p,
                                                       Picviz::PVAxis const& axis, uint32_t axis_index) :
	_sliders_manager_p(sm_p),
	_axis(&axis), _axis_index(axis_index)
{
	// This is needed to let the children of the group handle their events.
	setHandlesChildEvents(false);

	// the sliders must be over all other QGraphicsItems
	setZValue(1.e42);

	_sliders_group = new PVParallelView::PVSlidersGroup(sm_p, axis_index,
	                                                    this);

	addToGroup(_sliders_group);
	_sliders_group->setPos(PARALLELVIEW_AXIS_WIDTH / 2, 0.);
}

/*****************************************************************************
 * PVParallelView::PVAxisGraphicsItem::boundingRect
 *****************************************************************************/

QRectF PVParallelView::PVAxisGraphicsItem::boundingRect() const
{
	QRectF bbox = QRectF(
		-PVParallelView::AxisWidth*2,
		-PVAW_CST,
		PVParallelView::AxisWidth,
		IMAGE_HEIGHT + (2 * PVAW_CST)
	);

	return bbox.united(QRectF(- PVParallelView::AxisWidth, 0, 50, -50));
}

/*****************************************************************************
 * PVParallelView::PVAxisGraphicsItem::paint
 *****************************************************************************/

void PVParallelView::PVAxisGraphicsItem::paint(QPainter *painter,
                                               const QStyleOptionGraphicsItem *option,
                                               QWidget *widget)
{
	QPen pen = painter->pen();

	painter->fillRect(
		0,
		-PVAW_CST,
	    PVParallelView::AxisWidth,
	    IMAGE_HEIGHT + (2 * PVAW_CST),
	    _axis->get_color().toQColor()
	);
	painter->save();
	painter->translate(- PVParallelView::AxisWidth, - PVAW_CST);
	painter->rotate(-45.);
	painter->setPen(_axis->get_titlecolor().toQColor());
	painter->drawText(10, 0, _axis->get_name());
	painter->setPen(pen);
	painter->restore();

	QGraphicsItemGroup::paint(painter, option, widget);
}
