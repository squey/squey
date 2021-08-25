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

#include <pvparallelview/PVZoomedParallelViewSelectionLine.h>

#include <pvparallelview/PVZoomedParallelView.h>
#include <pvparallelview/PVSelectionGenerator.h>
#include <pvparallelview/PVSelectionRectangle.h>

#include <QPainter>
#include <QTimer>

/*****************************************************************************
 * PVParallelView::PVZoomedParallelViewSelectionLine::PVZoomedParallelViewSelectionLine
 *****************************************************************************/

PVParallelView::PVZoomedParallelViewSelectionLine::PVZoomedParallelViewSelectionLine(
    PVZoomedParallelView* zpv)
    : QGraphicsObject(nullptr), _zpv(zpv), _x_scale(1.0), _y_scale(1.0)
{
	_timer = new QTimer(this);
	_timer->setSingleShot(true);

	connect(_timer, &QTimer::timeout, this, &PVZoomedParallelViewSelectionLine::timeout);

	clear();
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelViewSelectionLine::~PVZoomedParallelViewSelectionLine
 *****************************************************************************/

PVParallelView::PVZoomedParallelViewSelectionLine::~PVZoomedParallelViewSelectionLine()
{
	delete _timer;
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelViewSelectionLine::boundingRect
 *****************************************************************************/

QRectF PVParallelView::PVZoomedParallelViewSelectionLine::boundingRect() const
{
	return QRectF(_tl_pos, _br_pos);
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelViewSelectionLine::paint
 *****************************************************************************/

void PVParallelView::PVZoomedParallelViewSelectionLine::paint(QPainter* painter,
                                                              const QStyleOptionGraphicsItem*,
                                                              QWidget*)
{
	const qreal line_offset = 10. / _x_scale;

	if (!isVisible()) {
		return;
	}

	painter->save();

	painter->setPen(QPen(_pen_color, 0));

	painter->drawLine(_tl_pos, _br_pos);

	painter->setPen(QPen(PVSelectionRectangle::handle_color, 0));

	qreal x1 = 0.;
	qreal x2 = _tl_pos.x();
	qreal y = _tl_pos.y();

	if (x2 < 0) {
		x2 -= line_offset;
	} else {
		x2 += line_offset;
	}

	painter->drawLine(QPointF(x1, y), QPointF(x2, y));

	x2 = _br_pos.x();
	y = _br_pos.y();

	if (x2 < 0) {
		x2 -= line_offset;
	} else {
		x2 += line_offset;
	}

	painter->drawLine(QPointF(x1, y), QPointF(x2, y));

	painter->restore();
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelViewSelectionLine::clear
 *****************************************************************************/

void PVParallelView::PVZoomedParallelViewSelectionLine::clear()
{
	_timer->stop();
	prepareGeometryChange();
	_tl_pos = QPointF();
	_br_pos = QPointF();
	hide();
	update();
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelViewSelectionLine::is_null
 *****************************************************************************/

bool PVParallelView::PVZoomedParallelViewSelectionLine::is_null() const
{
	return _tl_pos == _br_pos;
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelViewSelectionLine::top
 *****************************************************************************/

qreal PVParallelView::PVZoomedParallelViewSelectionLine::top() const
{
	return _tl_pos.y();
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelViewSelectionLine::bottom
 *****************************************************************************/

qreal PVParallelView::PVZoomedParallelViewSelectionLine::bottom() const
{
	return _br_pos.y();
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelViewSelectionLine::begin
 *****************************************************************************/

void PVParallelView::PVZoomedParallelViewSelectionLine::begin(const QPointF& p)
{
	start_timer();
	show();

	prepareGeometryChange();

	_ref_pos = p;
	_tl_pos = p;
	_br_pos = p;

	update();
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelViewSelectionLine::step
 *****************************************************************************/

void PVParallelView::PVZoomedParallelViewSelectionLine::step(const QPointF& p, bool need_timer)
{
	if (need_timer) {
		start_timer();
	}

	prepareGeometryChange();

	_tl_pos = QPointF(_ref_pos.x(), qMin(_ref_pos.y(), p.y()));
	_br_pos = QPointF(_ref_pos.x(), qMax(_ref_pos.y(), p.y()));

	update();
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelViewSelectionLine::end
 *****************************************************************************/

void PVParallelView::PVZoomedParallelViewSelectionLine::end(const QPointF& p)
{
	step(p, false);
	_timer->stop();
	hide();
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelViewSelectionLine::set_view_scale
 *****************************************************************************/

void PVParallelView::PVZoomedParallelViewSelectionLine::set_view_scale(const qreal xscale,
                                                                       const qreal yscale)
{
	_x_scale = xscale;
	_y_scale = yscale;
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelViewSelectionLine::start_timer
 *****************************************************************************/

void PVParallelView::PVZoomedParallelViewSelectionLine::start_timer()
{
	_pen_color = PVSelectionRectangle::handle_color;
	_timer->start(PVSelectionRectangle::delay_msec);
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelViewSelectionLine::timeout
 *****************************************************************************/

void PVParallelView::PVZoomedParallelViewSelectionLine::timeout()
{
	Q_EMIT commit_volatile_selection();

	_pen_color = PVSelectionRectangle::rectangle_color;
	get_zpview()->get_viewport()->update();
}
