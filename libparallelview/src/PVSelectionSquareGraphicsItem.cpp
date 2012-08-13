/**
 * \file PVSelectionSquareGraphicsItem.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvparallelview/PVSelectionSquareGraphicsItem.h>
#include <pvparallelview/PVFullParallelScene.h>

PVParallelView::PVSelectionSquareGraphicsItem::PVSelectionSquareGraphicsItem(PVFullParallelScene* s) :
	_volatile_selection_timer(new QTimer(this))
{
	_volatile_selection_timer->setSingleShot(true);
	connect(_volatile_selection_timer, SIGNAL(timeout()), this, SLOT(volatile_selection_timeout_Slot()));

	setZValue(std::numeric_limits<qreal>::max());
	if (s) {
		s->addItem(this);
	}
}
