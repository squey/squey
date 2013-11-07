
#include <pvparallelview/common.h>
#include <pvparallelview/PVZoomedSelectionAxisSlider.h>
#include <pvparallelview/PVSelectionRectangle.h>

#include <QPainter>

#define SLIDER_HALF_WIDTH 12
#define SLIDER_WIDTH (2 * SLIDER_HALF_WIDTH + PVParallelView::AxisWidth)
#define MARGIN 16

/*****************************************************************************
 * PVParallelView::PVZoomedSelectionAxisSlider::boundingRect
 *****************************************************************************/

QRectF PVParallelView::PVZoomedSelectionAxisSlider::boundingRect() const
{
	if (_orientation == Min) {
		return QRectF(QPointF(- SLIDER_HALF_WIDTH-MARGIN, 0),
		              QPointF(SLIDER_HALF_WIDTH+MARGIN, -4-MARGIN));
	} else {
		return QRectF(QPointF(- SLIDER_HALF_WIDTH-MARGIN, 0),
		              QPointF(SLIDER_HALF_WIDTH+MARGIN, 4+MARGIN));
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedSelectionAxisSlider::paint
 *****************************************************************************/

void PVParallelView::PVZoomedSelectionAxisSlider::paint(QPainter *painter,
                                                        const QStyleOptionGraphicsItem *option,
                                                        QWidget *widget)
{
	static const QRectF min_rect(QPointF(- SLIDER_HALF_WIDTH, 0),
	                             QPointF(SLIDER_HALF_WIDTH, -4));

	static const QRectF max_rect(QPointF(- SLIDER_HALF_WIDTH, 0),
	                             QPointF(SLIDER_HALF_WIDTH, 4));

	painter->save();

	QBrush b;

	if (mouse_is_hover()) {
		static const QColor c(PVSelectionRectangle::handle_color.red(),
		                      PVSelectionRectangle::handle_color.green(),
		                      PVSelectionRectangle::handle_color.blue(),
		                      PVSelectionRectangle::handle_transparency);

		b = QBrush(c, Qt::SolidPattern);
	} else {
		b = QBrush(Qt::black, Qt::SolidPattern);
	}

	if (_orientation == Min) {
		painter->fillRect(min_rect, b);
	} else {
		painter->fillRect(max_rect, b);
	}

	painter->setPen(PVSelectionRectangle::rectangle_color);

	if (_orientation == Min) {
		painter->drawRect(min_rect);
		painter->drawLine( SLIDER_HALF_WIDTH, 0,  SLIDER_HALF_WIDTH, 2);
		painter->drawLine(-SLIDER_HALF_WIDTH, 0, -SLIDER_HALF_WIDTH, 2);
	} else {
		painter->drawRect(max_rect);
		painter->drawLine( SLIDER_HALF_WIDTH, 0,  SLIDER_HALF_WIDTH, -1);
		painter->drawLine(-SLIDER_HALF_WIDTH, 0, -SLIDER_HALF_WIDTH, -1);
	}

	painter->restore();

	PVAbstractAxisSlider::paint(painter, option, widget);
}
