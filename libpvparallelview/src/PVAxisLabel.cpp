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

#include <squey/PVView.h>

#include <pvparallelview/PVAxisLabel.h>
#include <pvparallelview/PVAxisGraphicsItem.h>

#include <QDialog>
#include <QLayout>
#include <QMenu>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QFontMetrics>

#include <iostream>

/*****************************************************************************
 * PVParallelView::PVAxisLabel::PVAxisLabel
 *****************************************************************************/

PVParallelView::PVAxisLabel::PVAxisLabel(const Squey::PVView& view, QGraphicsItem* parent)
    : QGraphicsSimpleTextItem(parent), _lib_view(view)
{
	setFlag(QGraphicsItem::ItemClipsToShape, true);
}

/*****************************************************************************
 * PVParallelView::PVAxisLabel::~PVAxisLabel
 *****************************************************************************/

PVParallelView::PVAxisLabel::~PVAxisLabel()
{
	if (scene()) {
		scene()->removeItem(this);
	}
	if (group()) {
		group()->removeFromGroup(this);
	}
}

void PVParallelView::PVAxisLabel::set_bounding_box_width(int width)
{
	if (width) {
		_bounding_box_width = width;
	}
}

bool PVParallelView::PVAxisLabel::contains(const QPointF& point) const
{
	QRectF rect = QGraphicsSimpleTextItem::boundingRect();
	return rect.contains(point);
}

QRectF PVParallelView::PVAxisLabel::boundingRect() const
{
	return QGraphicsSimpleTextItem::boundingRect();
}

QPainterPath PVParallelView::PVAxisLabel::shape() const
{
	QPainterPath path;
	QRectF rect = QGraphicsSimpleTextItem::boundingRect();
	if (_bounding_box_width) {
		rect.setWidth(_bounding_box_width);
	}
	path.addRect(rect);

	return path;
}

PVParallelView::PVAxisGraphicsItem const* PVParallelView::PVAxisLabel::get_parent_axis() const
{
	return dynamic_cast<PVAxisGraphicsItem const*>(parentItem());
}

void PVParallelView::PVAxisLabel::set_text(const QString& text)
{
	QFontMetrics metrics = QFontMetrics(font());

	if (metrics.horizontalAdvance(text) > MAX_WIDTH) {

		setText(metrics.elidedText(text, Qt::ElideMiddle, MAX_WIDTH));

		setToolTip(text);
	} else {
		setText(text);
	}
}
