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

#include <pvparallelview/PVAbstractAxisSliders.h>
#include <pvparallelview/PVParallelView.h>

#include <QGraphicsSimpleTextItem>
#include <QPainter>

/*****************************************************************************
 * PVParallelView::PVAbstractAxisSliders::PVAbstractAxisSliders
 *****************************************************************************/

PVParallelView::PVAbstractAxisSliders::PVAbstractAxisSliders(QGraphicsItem* parent,
                                                             PVSlidersManager* sm_p,
                                                             PVSlidersGroup* group,
                                                             const char* text)
    : QObject(nullptr), QGraphicsItemGroup(parent), _sliders_manager_p(sm_p), _group(group)
{
	setHandlesChildEvents(false);

	/* RH: the 4 ' ' are a hacky visual offset to display the text at the
	 * left of the axis without overlapping the sliders
	 */
	_text = new QGraphicsSimpleTextItem(QString("    ") + text);
	addToGroup(_text);

	_text->setFlag(QGraphicsItem::ItemIgnoresTransformations, true);
	_text->setBrush(Qt::white);
	_text->hide();
}

void PVParallelView::PVAbstractAxisSliders::paint(QPainter* painter,
                                                  const QStyleOptionGraphicsItem* option,
                                                  QWidget* widget)
{
	QGraphicsItemGroup::paint(painter, option, widget);

#ifdef SQUEY_DEVELOPER_MODE
	if (common::show_bboxes()) {
		painter->save();
		painter->setPen(QPen(QColor(0xFF, 0xFF, 0), 0));
		painter->setBrush(QColor(0xFF, 0xFF, 0, 128));
		const QRectF br = boundingRect();
		painter->drawRect(br);
		painter->restore();
	}
#endif
}

QRectF PVParallelView::PVAbstractAxisSliders::boundingRect() const
{
	return childrenBoundingRect();
}
