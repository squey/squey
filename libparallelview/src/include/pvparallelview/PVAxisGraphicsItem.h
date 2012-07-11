/**
 * \file PVAxisGraphicsItem.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_PVAXISGRAPHICSITEM_H
#define PVPARALLELVIEW_PVAXISGRAPHICSITEM_H

#include <iostream>
#include <vector>
#include <utility>

#include <QGraphicsItem>

#include <pvparallelview/common.h>
#include <pvparallelview/PVAxisSlider.h>

// Used to draw the axis out of the image zone
#define PVAW_CST 8

namespace Picviz
{
class PVAxis;
}

namespace PVParallelView
{

typedef std::pair<PVAxisSlider*, PVAxisSlider*> PVAxisRangeSliders;

class PVAxisGraphicsItem : public QGraphicsItemGroup
{
public:
	PVAxisGraphicsItem(Picviz::PVAxis *axis);

	QRectF boundingRect () const;

	void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = 0);

	void add_range_sliders(uint32_t y1, uint32_t y2);
	bool sliders_moving() const;


	QRect map_from_scene(QRectF rect) const
	{
		QPointF point = mapFromScene(rect.topLeft());
		return QRect(point.x(), point.y(), rect.width(), rect.height());
	}

private:
	Picviz::PVAxis*                 _axis;
	QRectF                          _bbox;
	std::vector<PVAxisRangeSliders> _sliders;
};

}

#endif // PVPARALLELVIEW_PVAXISGRAPHICSITEM_H
