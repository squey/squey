
#ifndef PVPARALLELVIEW_PVABSTRACTAXISSLIDER_H
#define PVPARALLELVIEW_PVABSTRACTAXISSLIDER_H

#include <pvkernel/core/PVAlgorithms.h>

#include <QObject>
#include <QGraphicsItem>
#include <QGraphicsSceneContextMenuEvent>

/* TODO: move from int to uint32_t to use QPointF instead of QPoint to  move precisely
 *       any sliders in zoomed view
 */

namespace PVParallelView
{

class PVAbstractAxisSliders;

enum PVAxisSliderOrientation {
	Min,
	Max
};

class PVAbstractAxisSlider : public QGraphicsObject
{
Q_OBJECT

public:
	PVAbstractAxisSlider(int omin, int omax, int o,
	                     PVAxisSliderOrientation orientation = Min);

	~PVAbstractAxisSlider();

	void set_value(int v);

	inline int get_value() const
	{
		return _offset;
	}

	void set_range(int omin, int omax)
	{
		_offset_min = omin;
		_offset_max = omax;
	}

	void set_owner(PVAbstractAxisSliders *owner)
	{
		_owner = owner;
	}

	bool is_moving() const
	{
		return _moving;
	}

public:
	virtual QRectF boundingRect () const = 0;

	virtual void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = 0) = 0;

signals:
	void slider_moved();

protected:
	virtual void hoverEnterEvent(QGraphicsSceneHoverEvent * event);
	virtual void hoverMoveEvent(QGraphicsSceneHoverEvent * event);
	virtual void hoverLeaveEvent(QGraphicsSceneHoverEvent * event);

	virtual void mouseMoveEvent(QGraphicsSceneMouseEvent* event);
	virtual void mousePressEvent(QGraphicsSceneMouseEvent* event);
	virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent* event);

	virtual void contextMenuEvent(QGraphicsSceneContextMenuEvent *event);

	bool mouse_is_hover() const { return _is_hover; }

protected:
	int                     _offset_min;
	int                     _offset_max;
	int                     _offset;
	PVAxisSliderOrientation _orientation;
	bool                    _moving;
	bool                    _is_hover;
	PVAbstractAxisSliders  *_owner;
};

}

#endif // PVPARALLELVIEW_PVABSTRACTAXISSLIDER_H
