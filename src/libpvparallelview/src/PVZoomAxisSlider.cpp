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
#include <pvparallelview/PVZoomAxisSlider.h>

#include <QPainter>

#define ZOOM_LINE_WIDTH 12
#define ZOOM_ARROW_SIDE 6.0
#define MARGIN 16

/*****************************************************************************
 * PVParallelView::PVZoomAxisSlider::boundingRect
 *****************************************************************************/

QRectF PVParallelView::PVZoomAxisSlider::boundingRect() const
{
	if (_orientation == Min) {
		return QRectF(QPointF(-ZOOM_ARROW_SIDE - MARGIN, -MARGIN),
		              QPointF(ZOOM_ARROW_SIDE + MARGIN, 0));
	} else {
		return QRectF(QPointF(-ZOOM_ARROW_SIDE - MARGIN, 0),
		              QPointF(ZOOM_ARROW_SIDE + MARGIN, MARGIN));
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomAxisSlider::paint
 *****************************************************************************/

void PVParallelView::PVZoomAxisSlider::paint(QPainter* painter,
                                             const QStyleOptionGraphicsItem* option,
                                             QWidget* widget)
{
	static const QPointF min_points[3] = {
	    QPointF(0.0, 0.0), QPointF(-ZOOM_ARROW_SIDE, -ZOOM_ARROW_SIDE),
	    QPointF(ZOOM_ARROW_SIDE, -ZOOM_ARROW_SIDE),
	};

	static const QPointF max_points[3] = {
	    QPointF(0.0, 0.0), QPointF(-ZOOM_ARROW_SIDE, ZOOM_ARROW_SIDE),
	    QPointF(ZOOM_ARROW_SIDE, ZOOM_ARROW_SIDE),
	};

	painter->save();

	if (mouse_is_hover()) {
		painter->setBrush(QBrush(QColor(205, 56, 83, 192), Qt::SolidPattern));
	} else {
		painter->setBrush(QBrush(Qt::black, Qt::SolidPattern));
	}

	painter->setPen(QPen(Qt::white, 0));

	if (_orientation == Min) {
		painter->drawPolygon(min_points, 3);
	} else {
		painter->drawPolygon(max_points, 3);
	}

	painter->drawLine(QPointF(-ZOOM_LINE_WIDTH, 0), QPointF(ZOOM_LINE_WIDTH, 0));

	painter->restore();

	PVAbstractAxisSlider::paint(painter, option, widget);
}
