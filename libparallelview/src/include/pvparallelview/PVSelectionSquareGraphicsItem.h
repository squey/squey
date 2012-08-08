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

#include <pvparallelview/PVSelectionSquare.h>

namespace PVParallelView
{

class PVParallelScene;

class PVSelectionSquareGraphicsItem : public QGraphicsRectItem
{
public:
	PVSelectionSquareGraphicsItem(PVParallelScene* s);

	uint32_t compute_selection(PVZoneID zid, QRect rect, Picviz::PVSelection& sel)
	{
		return _selection_square->compute_selection(zid, rect, sel);
	}

	~PVSelectionSquareGraphicsItem()
	{
		delete _selection_square;
	}

	PVSelectionSquare* _selection_square;
};

}

#endif /* PVSELECTIONSQUAREGRAPHICSITEM_H_ */
