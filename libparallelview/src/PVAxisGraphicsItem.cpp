/**
 * \file PVAxisGraphicsItem.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <iostream>

#include <picviz/PVAxis.h>
#include <picviz/PVView.h>
#include <pvparallelview/PVAxisGraphicsItem.h>

#include <QPainter>
#include <QGraphicsScene>

// Used to draw the axis out of the image zone
#define PVAW_CST 8

/*****************************************************************************
 * PVParallelView::PVAxisGraphicsItem::PVAxisGraphicsItem
 *****************************************************************************/

PVParallelView::PVAxisGraphicsItem::PVAxisGraphicsItem(PVParallelView::PVSlidersManager_p sm_p,
                                                       Picviz::PVView const& view, const axe_id_t &axe_id) :
	_sliders_manager_p(sm_p),
	_axe_id(axe_id),
	_lib_view(view)
{
	// This is needed to let the children of the group handle their events.
	setHandlesChildEvents(false);

	// the sliders must be over all other QGraphicsItems
	setZValue(1.e42);

	_sliders_group = new PVSlidersGroup(sm_p, axe_id, this);

	addToGroup(get_sliders_group());
	get_sliders_group()->setPos(PARALLELVIEW_AXIS_WIDTH / 2, 0.);

	_label = new QGraphicsSimpleTextItem();
	addToGroup(_label);
	_label->rotate(-45.);
	_label->setPos(0, - 2 * PVAW_CST);

	update_axis_info();
}

PVParallelView::PVAxisGraphicsItem::~PVAxisGraphicsItem()
{
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
	    lib_axis()->get_color().toQColor()
	);
	painter->save();
	painter->translate(- PVParallelView::AxisWidth, - PVAW_CST);
	painter->rotate(-45.);
	painter->setPen(lib_axis()->get_titlecolor().toQColor());
	//painter->drawText(10, 0, lib_axis()->get_name());
	painter->setPen(pen);
	painter->restore();

	QGraphicsItemGroup::paint(painter, option, widget);
}

/*****************************************************************************
 * PVParallelView::PVAxisGraphicsItem::update_axis_info
 *****************************************************************************/

void PVParallelView::PVAxisGraphicsItem::update_axis_info()
{
	_label->setText(lib_axis()->get_name());
	_label->setBrush(lib_axis()->get_titlecolor().toQColor());
}

/*****************************************************************************
 * PVParallelView::PVAxisGraphicsItem::lib_axis
 *****************************************************************************/

Picviz::PVAxis const* PVParallelView::PVAxisGraphicsItem::lib_axis() const
{
	return &_lib_view.get_axis_by_id(_axe_id);
}
