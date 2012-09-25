
#ifndef PVPARALLELVIEW_PVABSTRACTAXISSLIDER_H
#define PVPARALLELVIEW_PVABSTRACTAXISSLIDER_H

#include <pvkernel/core/PVAlgorithms.h>

#include <QObject>
#include <QGraphicsItem>

/* TODO: move from int to uint32_t to use QPointF instead of QPoint to  move precisely
 *       any sliders in zoomed view
 */
namespace PVParallelView
{

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

	inline void set_value(int v)
	{
		_offset = PVCore::clamp(v, _offset_min, _offset_max);
		setPos(0., _offset);
	}

	inline int value()
	{
		return _offset;
	}

	void set_range(int omin, int omax)
	{
		_offset_min = omin;
		_offset_max = omax;
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
	void hoverEnterEvent(QGraphicsSceneHoverEvent * event);
	void hoverMoveEvent(QGraphicsSceneHoverEvent * event);
	void hoverLeaveEvent(QGraphicsSceneHoverEvent * event);

	void mouseMoveEvent(QGraphicsSceneMouseEvent* event);
	void mousePressEvent(QGraphicsSceneMouseEvent* event);
	void mouseReleaseEvent(QGraphicsSceneMouseEvent* event);

protected:
	int                     _offset_min;
	int                     _offset_max;
	int                     _offset;
	PVAxisSliderOrientation _orientation;
	bool                    _moving;
};

}

#endif // PVPARALLELVIEW_PVABSTRACTAXISSLIDER_H
