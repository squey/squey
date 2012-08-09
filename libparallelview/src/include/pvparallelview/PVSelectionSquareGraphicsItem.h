/**
 * \file PVSelectionSquareGraphicsItem.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVSELECTIONSQUAREGRAPHICSITEM_H_
#define PVSELECTIONSQUAREGRAPHICSITEM_H_

#include <limits>

#include <QGraphicsScene>
#include <QGraphicsRectItem>

namespace PVParallelView
{

class PVParallelScene;

class PVSelectionSquareGraphicsItem : public QGraphicsRectItem
{
public:
	PVSelectionSquareGraphicsItem(PVParallelScene* s);

	void clear_rect()
	{
		setRect(QRect());
	}
};

}

#endif /* PVSELECTIONSQUAREGRAPHICSITEM_H_ */
