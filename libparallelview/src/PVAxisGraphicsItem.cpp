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



/*****************************************************************************
 * PVParallelView::PVAxisGraphicsItem::PVAxisGraphicsItem
 *****************************************************************************/

PVParallelView::PVAxisGraphicsItem::PVAxisGraphicsItem(Picviz::PVAxis *axis, uint32_t axis_index) :
	_axis(axis), _axis_index(axis_index)
{
	setHandlesChildEvents(false); // This is needed to let the children of the group handle their events.
	setZValue(1.e42);
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
                                               const QStyleOptionGraphicsItem */*option*/,
                                               QWidget */*widget*/)
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
}

/*****************************************************************************
 * PVParallelView::PVAxisGraphicsItem::add_range_sliders
 *****************************************************************************/

void PVParallelView::PVAxisGraphicsItem::add_range_sliders(uint32_t p1, uint32_t p2)
{
	PVParallelView::PVAxisRangeSliders sliders;

	sliders.first = new PVParallelView::PVAxisSlider(0, PVParallelView::ImageHeight, p1);
	sliders.second = new PVParallelView::PVAxisSlider(0, PVParallelView::ImageHeight, p2);

	sliders.first->setPos(pos());
	sliders.second->setPos(pos());

	addToGroup(sliders.first);
	addToGroup(sliders.second);

	_sliders.push_back(sliders);

	// Connection
	connect(sliders.first, SIGNAL(slider_moved()), this, SLOT(slider_moved()));
	connect(sliders.second, SIGNAL(slider_moved()), this, SLOT(slider_moved()));
}

/*****************************************************************************
 * PVParallelView::PVAxisGraphicsItem::sliders_moving
 *****************************************************************************/

bool PVParallelView::PVAxisGraphicsItem::sliders_moving() const
{
	for (PVParallelView::PVAxisRangeSliders sliders : _sliders) {
		if (sliders.first->is_moving() || sliders.second->is_moving()) {
			return true;
		}
	}
	return false;
}
