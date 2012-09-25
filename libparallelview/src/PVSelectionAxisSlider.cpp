
#include <pvparallelview/common.h>
#include <pvparallelview/PVSelectionAxisSlider.h>

#include <QPainter>

#define SLIDER_HALF_WIDTH 10
#define SLIDER_WIDTH (2 * SLIDER_HALF_WIDTH + PVParallelView::AxisWidth)

/*****************************************************************************
 * PVParallelView::PVSelectionAxisSlider::boundingRect
 *****************************************************************************/

QRectF PVParallelView::PVSelectionAxisSlider::boundingRect() const
{
	if (_orientation == Min) {
		return QRectF(QPointF(- SLIDER_HALF_WIDTH, 0),
		              QPointF(SLIDER_HALF_WIDTH, -4));
	} else {
		return QRectF(QPointF(- SLIDER_HALF_WIDTH, 0),
		              QPointF(SLIDER_HALF_WIDTH, 4));
	}
}

/*****************************************************************************
 * PVParallelView::PVSelectionAxisSlider::paint
 *****************************************************************************/

void PVParallelView::PVSelectionAxisSlider::paint(QPainter *painter,
                                                  const QStyleOptionGraphicsItem *,
                                                  QWidget *)
{
	painter->fillRect(boundingRect(),
	                  Qt::black);

	QPen old_pen = painter->pen();
	painter->setPen(Qt::white);
	painter->drawRect(boundingRect());
	painter->setPen(old_pen);
}
