/**
 * \file PVSelectionRectangleItemItem.cpp
 *
 * Copyright (C) Picviz Labs 2013
 */

#include <pvparallelview/PVSelectionRectangleItem.h>
#include <pvparallelview/PVSelectionHandleItem.h>

#include <pvkernel/core/PVAlgorithms.h>

#include <QGraphicsScene>
#include <QGraphicsSceneHoverEvent>
#include <QPainter>

#include <iostream>

/*****************************************************************************
 * PVParallelView::PVSelectionRectangleItem::PVSelectionRectangleItem
 *****************************************************************************/

PVParallelView::PVSelectionRectangleItem::PVSelectionRectangleItem(const QRectF& rect,
                                                                   QGraphicsItem* parent) :
	QGraphicsObject(parent),
	_rect(rect),
	_x_min_value(0.),
	_x_max_value(0.),
	_y_min_value(0.),
	_y_max_value(0.)
{
	setAcceptHoverEvents(true);
	setHandlesChildEvents(false);

	_handles.push_back(new PVSelectionHandleItem(PVSelectionHandleItem::N, this));
	_handles.push_back(new PVSelectionHandleItem(PVSelectionHandleItem::NE, this));
	_handles.push_back(new PVSelectionHandleItem(PVSelectionHandleItem::E, this));
	_handles.push_back(new PVSelectionHandleItem(PVSelectionHandleItem::SE, this));
	_handles.push_back(new PVSelectionHandleItem(PVSelectionHandleItem::S, this));
	_handles.push_back(new PVSelectionHandleItem(PVSelectionHandleItem::SW, this));
	_handles.push_back(new PVSelectionHandleItem(PVSelectionHandleItem::W, this));
	_handles.push_back(new PVSelectionHandleItem(PVSelectionHandleItem::NW, this));
	_handles.push_back(new PVSelectionHandleItem(PVSelectionHandleItem::CENTER, this));

	set_pen_color(Qt::black);
	_pen.setCosmetic(true);
	_pen.setWidth(1.);

	QColor hc = Qt::gray;
	set_handles_pen_color(hc);
	hc.setAlpha(50);
	set_handles_brush_color(hc);

	update_handles();
	hide_all_handles();
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangleItem::clear
 *****************************************************************************/

void PVParallelView::PVSelectionRectangleItem::clear()
{
	prepareGeometryChange();
	_rect = QRectF();
	update_handles();
	hide();
	for(auto it : _handles) {
		it->hide();
		it->activate_cursor(false);
	}
	update();
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangleItem::begin
 *****************************************************************************/

void PVParallelView::PVSelectionRectangleItem::begin(const QPointF& p)
{
	qreal px = p.x();
	qreal py = p.y();

	if (_x_min_value != _x_max_value) {
		px = PVCore::clamp(px, _x_min_value, _x_max_value);
	}

	if (_y_min_value != _y_max_value) {
		py = PVCore::clamp(py, _y_min_value, _y_max_value);
	}

	QPointF np(px, py);

	_rect = QRectF(np, np);
	_ref = np;
	prepareGeometryChange();
	update_handles();
	show();
	for(auto it : _handles) {
		it->show();
	}
	set_handles_cursor(Qt::BlankCursor);
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangleItem::step
 *****************************************************************************/

void PVParallelView::PVSelectionRectangleItem::step(const QPointF& p)
{
	qreal px = p.x();
	qreal py = p.y();

	if (_x_min_value != _x_max_value) {
		px = PVCore::clamp(px, _x_min_value, _x_max_value);
	}

	if (_y_min_value != _y_max_value) {
		py = PVCore::clamp(py, _y_min_value, _y_max_value);
	}

	qreal nrl = std::min<qreal>(_ref.x(), px);
	qreal nrr = std::max<qreal>(_ref.x(), px);
	qreal nrt = std::min<qreal>(_ref.y(), py);
	qreal nrb = std::max<qreal>(_ref.y(), py);

	_rect = QRectF(QPointF(nrl, nrt), QPointF(nrr, nrb));
	prepareGeometryChange();
	update_handles();
}


/*****************************************************************************
 * PVParallelView::PVSelectionRectangleItem::end
 *****************************************************************************/

void PVParallelView::PVSelectionRectangleItem::end(const QPointF& p)
{
	reset_handles_cursor();
	step(p);
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangleItem::set_pen_color
 *****************************************************************************/

void PVParallelView::PVSelectionRectangleItem::set_pen_color(const QColor& col)
{
	_pen.setColor(col);
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangleItem::set_default_cursor
 *****************************************************************************/

void PVParallelView::PVSelectionRectangleItem::set_default_cursor(QCursor cursor)
{
	_default_cursor = cursor;
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangleItem::get_default_cursor
 *****************************************************************************/

QCursor PVParallelView::PVSelectionRectangleItem::get_default_cursor() const
{
	return _default_cursor;
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangleItem::set_handles_pen_color
 *****************************************************************************/

void PVParallelView::PVSelectionRectangleItem::set_handles_pen_color(const QColor& col) const
{
	for(auto it : _handles) {
		it->set_pen_color(col);
	}
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangleItem::set_handles_brush_color
 *****************************************************************************/

void PVParallelView::PVSelectionRectangleItem::set_handles_brush_color(const QColor& col) const
{
	for(auto it : _handles) {
		it->set_brush_color(col);
	}
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangleItem::set_handles_scale
 *****************************************************************************/

void PVParallelView::PVSelectionRectangleItem::set_handles_scale(const qreal xscale,
                                                                 const qreal yscale) const
{
	for(auto it : _handles) {
		it->set_scale(xscale, yscale);
	}
}


/*****************************************************************************
 * PVParallelView::PVSelectionRectangleItem::get_handles_x_scale
 *****************************************************************************/

qreal PVParallelView::PVSelectionRectangleItem::get_handles_x_scale() const
{
	// the scale factors is set for every handle, so we can pick anyone
	return _handles[0]->get_x_scale();
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangleItem::get_handles_y_scale
 *****************************************************************************/

qreal PVParallelView::PVSelectionRectangleItem::get_handles_y_scale() const
{
	// the scale factors is set for every handle, so we can pick anyone
	return _handles[0]->get_y_scale();
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangleItem::set_rect
 *****************************************************************************/

void PVParallelView::PVSelectionRectangleItem::set_rect(const QRectF& rect)
{
	prepareGeometryChange();

	qreal rl = rect.left();
	qreal rr = rect.right();
	qreal rt = rect.top();
	qreal rb = rect.bottom();

	bool need_hor_swap = (rl > rr);
	bool need_ver_swap = (rt > rb);

	qreal lc = std::min<qreal>(rl, rr);
	qreal rc = std::max<qreal>(rl, rr);
	qreal tc = std::min<qreal>(rt, rb);
	qreal bc = std::max<qreal>(rt, rb);

	QRectF old_rect = _rect;
	_rect = QRectF(QPointF(lc, tc), QPointF(rc, bc));

	for(auto it : _handles) {
		if (need_hor_swap) {
			it->swap_horizontally();
		}

		if (need_ver_swap) {
			it->swap_vertically();
		}
		it->update_geometry(_rect);
	}

	emit geometry_has_changed(old_rect, _rect);

	scene()->update(scene()->sceneRect());
	update();
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangleItem::get_rect
 *****************************************************************************/

QRectF PVParallelView::PVSelectionRectangleItem::get_rect()
{
	return _rect;
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangleItem::get_rect
 *****************************************************************************/

const QRectF PVParallelView::PVSelectionRectangleItem::get_rect() const
{
	return _rect;
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangleItem::set_x_range
 *****************************************************************************/

void PVParallelView::PVSelectionRectangleItem::set_x_range(qreal min_value,
                                                           qreal max_value)
{
	_x_min_value = min_value;
	_x_max_value = max_value;
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangleItem::set_y_range
 *****************************************************************************/

void PVParallelView::PVSelectionRectangleItem::set_y_range(qreal min_value,
                                                           qreal max_value)
{
	_y_min_value = min_value;
	_y_max_value = max_value;
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangleItem::clear_x_range
 *****************************************************************************/

void PVParallelView::PVSelectionRectangleItem::clear_x_range()
{
	_x_min_value = _x_max_value = 0.;
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangleItem::clear_y_range
 *****************************************************************************/

void PVParallelView::PVSelectionRectangleItem::clear_y_range()
{
	_y_min_value = _y_max_value = 0.;
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangleItem::boundingRect
 *****************************************************************************/

QRectF PVParallelView::PVSelectionRectangleItem::boundingRect() const
{
	return _rect;
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangleItem::paint
 *****************************************************************************/

void PVParallelView::PVSelectionRectangleItem::paint(QPainter* painter,
                                                     const QStyleOptionGraphicsItem* /*option*/,
                                                     QWidget* /*widget*/)
{
	painter->setPen(_pen);
	painter->drawRect(_rect);
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangleItem::itemChange
 *****************************************************************************/

QVariant PVParallelView::PVSelectionRectangleItem::itemChange(GraphicsItemChange change,
                                                              const QVariant& value)
{
	if (change == QGraphicsItem::ItemSceneHasChanged) {
		if (scene()) {
			for(const auto h : _handles) {
				scene()->addItem(h);
			}
		}
	} else if (change == QGraphicsItem::ItemZValueHasChanged) {
		qreal zvalue = zValue();
		for(const auto h : _handles) {
			h->setZValue(10. * zvalue);
		}
	}

	return QGraphicsItem::itemChange(change, value);
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangleItem::update_handles
 *****************************************************************************/

void PVParallelView::PVSelectionRectangleItem::update_handles() const
{
	for(auto it : _handles) {
		it->update_geometry(_rect);
	}
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangleItem::set_handles_cursor
 *****************************************************************************/

void PVParallelView::PVSelectionRectangleItem::set_handles_cursor(const QCursor& cursor)
{
	for(auto it : _handles) {
		it->setCursor(cursor);
	}
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangleItem::reset_handles_cursor
 *****************************************************************************/

void PVParallelView::PVSelectionRectangleItem::reset_handles_cursor()
{
	for(auto it : _handles) {
		it->activate_cursor(true);
	}
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangleItem::show_all_handles
 *****************************************************************************/

void PVParallelView::PVSelectionRectangleItem::show_all_handles() const
{
	for(auto it : _handles) {
		it->set_visible(true);
	}
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangleItem::hide_all_handles
 *****************************************************************************/

void PVParallelView::PVSelectionRectangleItem::hide_all_handles() const
{
	hide_all_handles_but(nullptr);
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangleItem::hide_all_handles_but
 *****************************************************************************/

void PVParallelView::PVSelectionRectangleItem::hide_all_handles_but(PVSelectionHandleItem* handle) const
{
	for(auto it : _handles) {
		if (it != handle) {
			it->set_visible(false);
		}
	}
}
