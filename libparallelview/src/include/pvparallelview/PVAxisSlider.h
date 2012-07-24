/**
 * \file PVAxisSlider.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_PVAXISSLIDER_H
#define PVPARALLELVIEW_PVAXISSLIDER_H

#include <QGraphicsItem>

namespace PVParallelView
{

class PVAxisSlider : public QGraphicsItem
{
public:
	PVAxisSlider(int omin, int omax, int o);
	~PVAxisSlider();

	QRectF boundingRect () const;

	void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = 0);

private:
	void hoverEnterEvent(QGraphicsSceneHoverEvent * event);
	void hoverMoveEvent(QGraphicsSceneHoverEvent * event);
	void hoverLeaveEvent(QGraphicsSceneHoverEvent * event);

private:
	int _offset_min, _offset_max;
	int _offset;
};

}

#endif // PVPARALLELVIEW_PVAXISSLIDER_H
