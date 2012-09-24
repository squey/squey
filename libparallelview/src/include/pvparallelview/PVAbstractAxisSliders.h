
#ifndef PVPARALLELVIEW_PVABSTRACTAXISSLIDERS_H
#define PVPARALLELVIEW_PVABSTRACTAXISSLIDERS_H

#include <QObject>
#include <QGraphicsItemGroup>
#include <QPainter>

#include <iostream>

namespace PVParallelView
{

class PVSlidersGroup;

class PVAbstractAxisSliders : public QObject, public QGraphicsItemGroup
{
public:
	PVAbstractAxisSliders(QGraphicsItem *parent, PVSlidersGroup *group) :
		QObject(nullptr), QGraphicsItemGroup(parent),
		_group(group)
	{}

	virtual bool is_moving() const = 0;

protected:
	PVSlidersGroup *_group;
};

}

#endif // PVPARALLELVIEW_PVABSTRACTAXISSLIDERS_H
