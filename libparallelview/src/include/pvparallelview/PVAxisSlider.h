/**
 * \file PVAxisSlider.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_PVAXISSLIDER_H
#define PVPARALLELVIEW_PVAXISSLIDER_H

#include <QGraphicsItem>

#include <pvparallelview/common.h>

#define SLIDER_HALF_WIDTH 8
#define SLIDER_WIDTH (2 * SLIDER_HALF_WIDTH + PVParallelView::AxisWidth)

namespace PVParallelView
{

enum {
	SliderHalfWidth = SLIDER_HALF_WIDTH,
	SliderWidth = SLIDER_WIDTH
};

class PVAxisSlider : public QObject, public QGraphicsItem
{
	Q_OBJECT

public:
	PVAxisSlider(int omin, int omax, int o);
	~PVAxisSlider();

	inline int value() { return _offset; }

	QRectF boundingRect () const;
	bool is_moving() const;

	void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = 0);

protected:
	void hoverEnterEvent(QGraphicsSceneHoverEvent * event);
	void hoverMoveEvent(QGraphicsSceneHoverEvent * event);
	void hoverLeaveEvent(QGraphicsSceneHoverEvent * event);

	void mouseMoveEvent(QGraphicsSceneMouseEvent* event);
	void mousePressEvent(QGraphicsSceneMouseEvent* event);
	void mouseReleaseEvent(QGraphicsSceneMouseEvent* event);

signals:
	void slider_moved();

private:
	int _offset_min, _offset_max;
	int _offset;
	bool _moving;
};

}

#endif // PVPARALLELVIEW_PVAXISSLIDER_H
