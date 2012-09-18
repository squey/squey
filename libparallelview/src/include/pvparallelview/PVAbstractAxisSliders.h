
#ifndef PVPARALLELVIEW_PVABSTRACTAXISSLIDERS_H
#define PVPARALLELVIEW_PVABSTRACTAXISSLIDERS_H

#include <QObject>
#include <QGraphicsItemGroup>
#include <QPainter>

#include <iostream>

namespace PVParallelView
{

class PVAbstractAxisSliders : public QObject, public QGraphicsItemGroup
{
public:
	PVAbstractAxisSliders(QGraphicsItem *parent) : QObject(nullptr), QGraphicsItemGroup(parent)
	{}

	virtual bool is_moving() const = 0;

private:
};

}

#endif // PVPARALLELVIEW_PVABSTRACTAXISSLIDERS_H
