/**
 * \file PVAxisGraphicsItem.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <iostream>

#include <picviz/PVAxis.h>
#include <picviz/PVView.h>

#include <pvparallelview/PVAxisGraphicsItem.h>
#include <pvparallelview/PVAxisLabel.h>

#include <QPainter>
#include <QGraphicsScene>

/*****************************************************************************
 * PVParallelView::PVAxisGraphicsItem::PVAxisGraphicsItem
 *****************************************************************************/

PVParallelView::PVAxisGraphicsItem::PVAxisGraphicsItem(PVParallelView::PVSlidersManager_p sm_p,
                                                       Picviz::PVView const& view, const axis_id_t &axis_id) :
	_sliders_manager_p(sm_p),
	_axis_id(axis_id),
	_lib_view(view)
{
	// This is needed to let the children of the group handle their events.
	setHandlesChildEvents(false);

	// the sliders must be over all other QGraphicsItems
	setZValue(1.e42);

	_sliders_group = new PVSlidersGroup(sm_p, axis_id, this);

	addToGroup(get_sliders_group());
	get_sliders_group()->setPos(PARALLELVIEW_AXIS_WIDTH / 2, 0.);

	_label = new PVAxisLabel(view, _sliders_group);
	addToGroup(_label);
	_label->rotate(-45.);
	_label->setPos(0, - 2 * axis_extend);

	connect(_label, SIGNAL(new_zoomed_parallel_view(int)), this, SLOT(emit_new_zoomed_parallel_view(int)));

	update_axis_info();
}

PVParallelView::PVAxisGraphicsItem::~PVAxisGraphicsItem()
{
}

/*****************************************************************************
 * PVParallelView::PVAxisGraphicsItem::paint
 *****************************************************************************/

void PVParallelView::PVAxisGraphicsItem::paint(QPainter *painter,
                                               const QStyleOptionGraphicsItem *option,
                                               QWidget *widget)
{
	painter->fillRect(
		0,
		-axis_extend,
	    PVParallelView::AxisWidth,
	    _axis_length + (2 * axis_extend),
	    lib_axis()->get_color().toQColor()
	);

	QGraphicsItemGroup::paint(painter, option, widget);
}

/*****************************************************************************
 * PVParallelView::PVAxisGraphicsItem::update_axis_info
 *****************************************************************************/

void PVParallelView::PVAxisGraphicsItem::update_axis_info()
{
	_label->set_text(lib_axis()->get_name());
	_label->set_color(lib_axis()->get_titlecolor().toQColor());
	_label->set_axis_index(_lib_view.get_axes_combination().get_index_by_id(_axis_id));
}

/*****************************************************************************
 * PVParallelView::PVAxisGraphicsItem::get_label_scene_bbox
 *****************************************************************************/

QRectF PVParallelView::PVAxisGraphicsItem::get_label_scene_bbox() const
{
	return _label->get_scene_bbox();
}

/*****************************************************************************
 * PVParallelView::PVAxisGraphicsItem::lib_axis
 *****************************************************************************/

Picviz::PVAxis const* PVParallelView::PVAxisGraphicsItem::lib_axis() const
{
	return &_lib_view.get_axis_by_id(_axis_id);
}
