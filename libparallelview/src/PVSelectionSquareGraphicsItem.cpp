/**
 * \file PVSelectionSquareGraphicsItem.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvparallelview/PVSelectionSquareGraphicsItem.h>
#include <pvparallelview/PVParallelScene.h>

PVParallelView::PVSelectionSquareGraphicsItem::PVSelectionSquareGraphicsItem(PVParallelScene* s)
{
	setPen(QPen(Qt::red, 2));
	setZValue(std::numeric_limits<qreal>::max());
	if (s) {
		s->addItem(this);
	}
}
