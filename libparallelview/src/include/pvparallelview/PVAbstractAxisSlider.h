
#ifndef PVPARALLELVIEW_PVABSTRACTAXISSLIDER_H
#define PVPARALLELVIEW_PVABSTRACTAXISSLIDER_H

#include <pvkernel/core/PVAlgorithms.h>

#include <pvparallelview/common.h>

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
	constexpr static int64_t min_value = 0LL;
	constexpr static int64_t max_value = (1LL << 32);
	constexpr static int64_t precision = (1LL << (32 - NBITS_INDEX));


public:
	PVAbstractAxisSlider(int64_t omin, int64_t omax, int64_t o,
	                     PVAxisSliderOrientation orientation = Min);

	~PVAbstractAxisSlider();

	void set_value(int64_t v);

	inline int64_t get_value() const
	{
		return _offset;
	}

	void set_range(int64_t omin, int64_t omax)
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

protected:
	int64_t                 _offset_min;
	int64_t                 _offset_max;
	int64_t                 _offset;
	PVAxisSliderOrientation _orientation;
	bool                    _moving;
	PVAbstractAxisSliders  *_owner;
};

}

#endif // PVPARALLELVIEW_PVABSTRACTAXISSLIDER_H
