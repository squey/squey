/**
 * \file PVSelectionHandleItem.cpp
 *
 * Copyright (C) Picviz Labs 2013
 */

#include <pvparallelview/PVSelectionHandleItem.h>
#include <pvparallelview/PVSelectionRectangleItem.h>

#include <QPainter>
#include <QPaintEngine>
#include <QGraphicsSceneMouseEvent>

/**
 * @todo: voir pourquoi les curseurs ne marchent plus
 */

/**
 * to invert handle's type when the selection rectangle is "reverted" while
 * moving a handle.
 */
static int masked_inversion(int value, int mask)
{
	int unmasked = value & ~mask;
	int masked = value & mask;
	if (masked != 0) {
		// bits must be inverted only if there is at least one bit set
		masked ^= mask;
	}

	int res = unmasked | masked;

	return res;
}

/*****************************************************************************
 * PVParallelView::PVSelectionHandleItem::PVSelectionHandleItem
 *****************************************************************************/

PVParallelView::PVSelectionHandleItem::PVSelectionHandleItem(handle_type type,
                                                             PVSelectionRectangleItem* sel_rect) :
	QGraphicsItem(nullptr),
	_sel_rect(sel_rect),
	_xscale(1.0),
	_yscale(1.0),
	_type(type),
	_is_visible(true)
{
	setAcceptHoverEvents(true);

	set_pen_color(Qt::black);
	set_brush_color(QColor(127, 127, 127, 50));

	_pen.setCosmetic(true);
	_pen.setWidth(1.0);
	update_geometry(QRectF());
}

/*****************************************************************************
 * PVParallelView::PVSelectionHandleItem::set_pen_color
 *****************************************************************************/

void PVParallelView::PVSelectionHandleItem::set_pen_color(QColor col)
{
	_pen.setColor(col);
}

/*****************************************************************************
 * PVParallelView::PVSelectionHandleItem::set_brush_color
 *****************************************************************************/

void PVParallelView::PVSelectionHandleItem::set_brush_color(QColor col)
{
	_brush_color = col;
}

/*****************************************************************************
 * PVParallelView::PVSelectionHandleItem::set_scale
 *****************************************************************************/

void PVParallelView::PVSelectionHandleItem::set_scale(const qreal xscale,
                                                      const qreal yscale)
{
	_xscale = xscale;
	_yscale = yscale;
	update_geometry(get_selection_rectangle()->boundingRect());
}

/*****************************************************************************
 * PVParallelView::PVSelectionHandleItem::boundingRect
 *****************************************************************************/

QRectF PVParallelView::PVSelectionHandleItem::boundingRect() const
{
	return _rect;
}

/*****************************************************************************
 * PVParallelView::PVSelectionHandleItem::paint
 *****************************************************************************/

void PVParallelView::PVSelectionHandleItem::paint(QPainter* painter,
                                                  const QStyleOptionGraphicsItem* /*option*/,
                                                  QWidget* /*widget*/)
{
	if (!is_visible()) {
		return;
	}

	painter->save();
	painter->setPen(_pen);
	painter->setBrush(_brush);
	painter->drawRect(_rect);
	painter->restore();
}

/*****************************************************************************
 * PVParallelView::PVSelectionHandleItem::hoverEnterEvent
 *****************************************************************************/

void PVParallelView::PVSelectionHandleItem::hoverEnterEvent(QGraphicsSceneHoverEvent* /*event*/)
{
	_brush.setStyle(Qt::SolidPattern);
	_brush.setColor(_brush_color);
	get_selection_rectangle()->show_all_handles();
	update();
}

/*****************************************************************************
 * PVParallelView::PVSelectionHandleItem::hoverLeaveEvent
 *****************************************************************************/

void PVParallelView::PVSelectionHandleItem::hoverLeaveEvent(QGraphicsSceneHoverEvent* /*event*/)
{
	_brush.setStyle(Qt::NoBrush);
	get_selection_rectangle()->hide_all_handles();
	update();
}

/*****************************************************************************
 * PVParallelView::PVSelectionHandleItem::mousePressEvent
 *****************************************************************************/

void PVParallelView::PVSelectionHandleItem::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
	if (event->button() == Qt::LeftButton) {
		_ref = event->pos();
		get_selection_rectangle()->hide_all_handles_but(this);
		get_selection_rectangle()->set_handles_cursor(Qt::OpenHandCursor);
		event->accept();
	}
}

/*****************************************************************************
 * PVParallelView::PVSelectionHandleItem::mouseReleaseEvent
 *****************************************************************************/

void PVParallelView::PVSelectionHandleItem::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
	if (event->button() == Qt::LeftButton) {
		update_selection_rectangle_geometry(event->pos());
		get_selection_rectangle()->show_all_handles();
		get_selection_rectangle()->reset_handles_cursor();
		event->accept();
		activate_cursor(true);
	}
}

/*****************************************************************************
 * PVParallelView::PVSelectionHandleItem::mouseMoveEvent
 *****************************************************************************/

void PVParallelView::PVSelectionHandleItem::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
	if (event->buttons() == Qt::LeftButton) {
		update_selection_rectangle_geometry(event->pos());
		event->accept();
	} else {
		QGraphicsItem::mouseMoveEvent(event);
	}
}

/*****************************************************************************
 * PVParallelView::PVSelectionHandleItem::get_selection_rectangle
 *****************************************************************************/

PVParallelView::PVSelectionRectangleItem* PVParallelView::PVSelectionHandleItem::get_selection_rectangle()
{
	return _sel_rect;
}

/*****************************************************************************
 * PVParallelView::PVSelectionHandleItem::get_selection_rectangle
 *****************************************************************************/

const PVParallelView::PVSelectionRectangleItem* PVParallelView::PVSelectionHandleItem::get_selection_rectangle() const
{
	return _sel_rect;
}

/*****************************************************************************
 * PVParallelView::PVSelectionHandleItem::update_selection_rectangle_geometry
 *****************************************************************************/

void PVParallelView::PVSelectionHandleItem::update_selection_rectangle_geometry(const QPointF& p)
{
	QPointF delta = p - _ref;

	switch(_type) {
	case N:
	case S:
		delta.setX(0.);
		break;
	case W:
	case E:
		delta.setY(0.);
		break;
	default:
		break;
	}

	setPos(pos() + delta);

	PVSelectionRectangleItem* parent = get_selection_rectangle();
	QRectF rect = parent->get_rect();

	switch(_type) {
	case N:
		rect.setTopLeft(pos());
		break;
	case NE:
		rect.setTopRight(pos());
		break;
	case E:
		rect.setTopRight(pos());
		break;
	case SE:
		rect.setBottomRight(pos());
		break;
	case S:
		rect.setBottomLeft(pos());
		break;
	case SW:
		rect.setBottomLeft(pos());
		break;
	case W:
		rect.setTopLeft(pos());
		break;
	case NW:
		rect.setTopLeft(pos());
		break;
	case CENTER:
		rect.moveTopLeft(pos());
		break;
	}

	parent->set_rect(rect);
}

/*****************************************************************************
 * PVParallelView::PVSelectionHandleItem::update_geometry
 *****************************************************************************/

void PVParallelView::PVSelectionHandleItem::update_geometry(const QRectF& rect)
{
	prepareGeometryChange();
	qreal h_xsize = handle_size * _xscale;
	qreal h_ysize = handle_size * _yscale;

	switch(_type) {
	case N:
		_rect = QRectF(0, -h_ysize,
		               rect.width(), h_ysize);
		setPos(rect.topLeft());
		break;
	case NE:
		_rect = QRectF(0, -h_ysize,
		               h_xsize, h_ysize);
		setPos(rect.topRight());
		break;
	case E:
		_rect = QRectF(0, 0,
		               h_xsize, rect.height());
		setPos(rect.topRight());
		break;
	case SE:
		_rect = QRectF(0, 0,
		               h_xsize, h_ysize);
		setPos(rect.bottomRight());
		break;
	case S:
		_rect = QRectF(0, 0,
		               rect.width(), h_ysize);
		setPos(rect.bottomLeft());
		break;
	case SW:
		_rect = QRectF(-h_xsize, 0,
		               h_xsize, h_ysize);
		setPos(rect.bottomLeft());
		break;
	case W:
		_rect = QRectF(-h_xsize, 0,
		               h_xsize, rect.height());
		setPos(rect.topLeft());
		break;
	case NW:
		_rect = QRectF(-h_xsize, -h_ysize,
		               h_xsize, h_ysize);
		setPos(rect.topLeft());
		break;
	case CENTER:
		_rect = QRectF(0, 0,
		               rect.width(), rect.height());
		setPos(rect.topLeft());
		break;
	}

	update();
}

/*****************************************************************************
 * PVParallelView::PVSelectionHandleItem::swap_horizontally
 *****************************************************************************/

void PVParallelView::PVSelectionHandleItem::swap_horizontally()
{
	if (_type != CENTER) {
		_type = masked_inversion(_type, E | W);
	}
}

/*****************************************************************************
 * PVParallelView::PVSelectionHandleItem::swap_vertically
 *****************************************************************************/

void PVParallelView::PVSelectionHandleItem::swap_vertically()
{
	if (_type != CENTER) {
		_type = masked_inversion(_type, N | S);
	}
}

/*****************************************************************************
 * PVParallelView::PVSelectionHandleItem::set_visible
 *****************************************************************************/

void PVParallelView::PVSelectionHandleItem::set_visible(bool visible)
{
	_is_visible = visible;
	update();
}

/*****************************************************************************
 * PVParallelView::PVSelectionHandleItem::is_visible
 *****************************************************************************/

bool PVParallelView::PVSelectionHandleItem::is_visible() const
{
	return _is_visible;
}

/*****************************************************************************
 * PVParallelView::PVSelectionHandleItem::activate_cursor
 *****************************************************************************/

void PVParallelView::PVSelectionHandleItem::activate_cursor(bool use_own)
{
	QCursor cursor;

	if (use_own) {
		switch(_type) {
		case N:
			cursor = Qt::SizeVerCursor;
			break;
		case NE:
			cursor = Qt::SizeBDiagCursor;
			break;
		case E:
			cursor = Qt::SizeHorCursor;
			break;
		case SE:
			cursor = Qt::SizeFDiagCursor;
			break;
		case S:
			cursor = Qt::SizeVerCursor;
			break;
		case SW:
			cursor = Qt::SizeBDiagCursor;
			break;
		case W:
			cursor = Qt::SizeHorCursor;
			break;
		case NW:
			cursor = Qt::SizeFDiagCursor;
			break;
		case CENTER:
			cursor = Qt::OpenHandCursor;
			break;
		}
	} else {
		cursor = get_selection_rectangle()->get_default_cursor();
	}

	setCursor(cursor);
}

