
#include <pvparallelview/common.h>
#include <pvparallelview/PVZoomAxisSlider.h>

#include <QPainter>

#define ZOOM_LINE_WIDTH 12
#define ZOOM_ARROW_SIDE 6.0
#define MARGIN 25

/*****************************************************************************
 * PVParallelView::PVZoomAxisSlider::boundingRect
 *****************************************************************************/

QRectF PVParallelView::PVZoomAxisSlider::boundingRect() const
{
	if (_orientation == Min) {
		return QRectF(QPointF(-ZOOM_ARROW_SIDE-MARGIN, -MARGIN),
		              QPointF(ZOOM_ARROW_SIDE+MARGIN, ZOOM_ARROW_SIDE+MARGIN));
	} else {
		return QRectF(QPointF(-ZOOM_ARROW_SIDE-MARGIN, MARGIN),
		              QPointF(ZOOM_ARROW_SIDE+MARGIN, -ZOOM_ARROW_SIDE-MARGIN));
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomAxisSlider::paint
 *****************************************************************************/

void PVParallelView::PVZoomAxisSlider::paint(QPainter *painter,
                                             const QStyleOptionGraphicsItem *,
                                             QWidget *)
{
	static const QPointF min_points[3] = {
		QPointF(             0.0,             0.0),
		QPointF(-ZOOM_ARROW_SIDE, ZOOM_ARROW_SIDE),
		QPointF( ZOOM_ARROW_SIDE, ZOOM_ARROW_SIDE),
	};

	static const QPointF max_points[3] = {
		QPointF(             0.0,              0.0),
		QPointF(-ZOOM_ARROW_SIDE, -ZOOM_ARROW_SIDE),
		QPointF( ZOOM_ARROW_SIDE, -ZOOM_ARROW_SIDE),
	};

	painter->save();

	if (mouse_is_hover()) {
		painter->setBrush(QBrush(QColor(205, 56, 83),Qt::SolidPattern));
	}
	else {
		painter->setBrush(QBrush(Qt::black,Qt::SolidPattern));
	}
	painter->setPen(Qt::white);

	if (_orientation == Min) {
		painter->drawPolygon(min_points, 3);
	} else {
		painter->drawPolygon(max_points, 3);
	}
	painter->drawLine(QPointF(-ZOOM_LINE_WIDTH, 0),
	                  QPointF(ZOOM_LINE_WIDTH, 0));

	painter->restore();
}
