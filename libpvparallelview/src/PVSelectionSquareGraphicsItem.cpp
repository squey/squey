/**
 * \file PVSelectionSquareGraphicsItem.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvparallelview/PVSelectionSquareGraphicsItem.h>

#include <QDebug>
#include <QPainter>
#include <QRect>
#include <QStyleOptionGraphicsItem>

PVParallelView::PVSelectionSquareGraphicsItem::PVSelectionSquareGraphicsItem(QGraphicsItem* parent):
	QGraphicsRectItem(parent),
	_volatile_selection_timer(new QTimer(this))
{
	_volatile_selection_timer->setSingleShot(true);
	connect(_volatile_selection_timer, SIGNAL(timeout()), this, SLOT(volatile_selection_timeout_Slot()));

	// lines thickness must not be affected by view's transformation
	QPen cur_pen = pen();
	cur_pen.setCosmetic(true);
	cur_pen.setWidth(PEN_WIDTH);
	setPen(cur_pen);

	setZValue(std::numeric_limits<qreal>::max());
}

void PVParallelView::PVSelectionSquareGraphicsItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* /*option*/, QWidget* /*widget*/)
{
	// Fix for OpenGL with "big" scale transformation
	QTransform const& pt = painter->transform();
	const QRectF view_rect = pt.mapRect(rect());

	painter->save();
	painter->resetTransform();
	painter->setPen(pen());
	painter->setBrush(brush());
	painter->drawRect(view_rect);
	painter->restore();
}
