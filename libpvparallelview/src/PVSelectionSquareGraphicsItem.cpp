/**
 * \file PVSelectionSquareGraphicsItem.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvparallelview/PVSelectionSquareGraphicsItem.h>

PVParallelView::PVSelectionSquareGraphicsItem::PVSelectionSquareGraphicsItem(QGraphicsScene* s) :
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
	if (s) {
		s->addItem(this);
	}
}
