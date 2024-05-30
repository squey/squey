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

#include <pvparallelview/PVAbstractRangeAxisSliders.h>
#include <pvparallelview/PVSlidersGroup.h>

#include <QPainter>
#include <QGraphicsScene>

/*****************************************************************************
 * PVParallelView::PVAbstractRangeAxisSliders::PVAbstractRangeAxisSliders
 *****************************************************************************/

PVParallelView::PVAbstractRangeAxisSliders::PVAbstractRangeAxisSliders(
    QGraphicsItem* parent,
    PVSlidersManager* sm_p,
    PVParallelView::PVSlidersGroup* group,
    const char* text)
    : PVAbstractAxisSliders(parent, sm_p, group, text), _sl_min(nullptr), _sl_max(nullptr)
{
}

/*****************************************************************************
 * PVParallelView::PVAbstractRangeAxisSliders::~PVAbstractRangeAxisSliders
 *****************************************************************************/

PVParallelView::PVAbstractRangeAxisSliders::~PVAbstractRangeAxisSliders()
{
	if (_sl_min) {
		if (scene()) {
			scene()->removeItem(_sl_min);
		}
		removeFromGroup(_sl_min);
		delete _sl_min;
	}
	if (_sl_max) {
		if (scene()) {
			scene()->removeItem(_sl_max);
		}
		removeFromGroup(_sl_max);
		delete _sl_max;
	}
}

/*****************************************************************************
 * PVParallelView::PVAbstractRangeAxisSliders::paint
 *****************************************************************************/

void PVParallelView::PVAbstractRangeAxisSliders::paint(QPainter* painter,
                                                       const QStyleOptionGraphicsItem* option,
                                                       QWidget* widget)
{
	if (_sl_min and _sl_max) {
		qreal vmin = _sl_min->pos().y();
		_text->setPos(0, vmin);

		if (is_moving()) {
			painter->save();

			painter->setCompositionMode(QPainter::RasterOp_SourceXorDestination);

			painter->setPen(QPen(Qt::white, 0));
			qreal vmax = _sl_max->pos().y();
			painter->drawLine(QPointF(0., vmin), QPointF(0., vmax));

			_text->show();

			painter->restore();
		} else {
			_text->hide();
		}
	}

	PVAbstractAxisSliders::paint(painter, option, widget);
}
