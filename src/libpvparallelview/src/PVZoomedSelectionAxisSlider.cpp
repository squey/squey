//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvparallelview/common.h>
#include <pvparallelview/PVZoomedSelectionAxisSlider.h>
#include <pvparallelview/PVSelectionRectangle.h>

#include <QPainter>

#define SLIDER_HALF_WIDTH 12
#define MARGIN 16

/*****************************************************************************
 * PVParallelView::PVZoomedSelectionAxisSlider::boundingRect
 *****************************************************************************/

QRectF PVParallelView::PVZoomedSelectionAxisSlider::boundingRect() const
{
	if (_orientation == Min) {
		return QRectF(QPointF(-SLIDER_HALF_WIDTH - MARGIN, 0),
		              QPointF(SLIDER_HALF_WIDTH + MARGIN, -4 - MARGIN));
	} else {
		return QRectF(QPointF(-SLIDER_HALF_WIDTH - MARGIN, 0),
		              QPointF(SLIDER_HALF_WIDTH + MARGIN, 4 + MARGIN));
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedSelectionAxisSlider::paint
 *****************************************************************************/

void PVParallelView::PVZoomedSelectionAxisSlider::paint(QPainter* painter,
                                                        const QStyleOptionGraphicsItem* option,
                                                        QWidget* widget)
{
	static const QRectF min_rect(QPointF(-SLIDER_HALF_WIDTH, 0), QPointF(SLIDER_HALF_WIDTH, -4));

	static const QRectF max_rect(QPointF(-SLIDER_HALF_WIDTH, 0), QPointF(SLIDER_HALF_WIDTH, 4));

	painter->save();

	QBrush b;

	if (mouse_is_hover()) {
		static const QColor c(
		    PVSelectionRectangle::handle_color.red(), PVSelectionRectangle::handle_color.green(),
		    PVSelectionRectangle::handle_color.blue(), PVSelectionRectangle::handle_transparency);

		b = QBrush(c, Qt::SolidPattern);
	} else {
		b = QBrush(Qt::black, Qt::SolidPattern);
	}

	if (_orientation == Min) {
		painter->fillRect(min_rect, b);
	} else {
		painter->fillRect(max_rect, b);
	}

	painter->setPen(QPen(PVSelectionRectangle::rectangle_color, 0));

	if (_orientation == Min) {
		painter->drawRect(min_rect);
		painter->drawLine(SLIDER_HALF_WIDTH, 0, SLIDER_HALF_WIDTH, 2);
		painter->drawLine(-SLIDER_HALF_WIDTH, 0, -SLIDER_HALF_WIDTH, 2);
	} else {
		painter->drawRect(max_rect);
		painter->drawLine(SLIDER_HALF_WIDTH, 0, SLIDER_HALF_WIDTH, -1);
		painter->drawLine(-SLIDER_HALF_WIDTH, 0, -SLIDER_HALF_WIDTH, -1);
	}

	painter->restore();

	PVAbstractAxisSlider::paint(painter, option, widget);
}
