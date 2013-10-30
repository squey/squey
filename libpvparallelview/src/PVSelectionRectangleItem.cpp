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


#define print_r(R) __print_rect(#R, R)
#define print_rect(R) __print_rect(#R, R)

template <typename R>
void __print_rect(const char *text, const R &r)
{
	std::cout << text << ": "
	          << r.x() << " " << r.y() << ", "
	          << r.width() << " " << r.height()
	          << std::endl;
}

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
	_y_max_value(0.),
	_sel_mode(RECTANGLE)
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
	_central_handle = new PVSelectionHandleItem(PVSelectionHandleItem::CENTER, this);
	_handles.push_back(_central_handle);

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

	qreal px2 = px;
	qreal py2 = py;

	if (_sel_mode == HORIZONTAL) {
		px = _x_min_value;
		px2 = _x_max_value;
	} else if (_sel_mode == VERTICAL) {
		py = _y_min_value;
		py2 = _y_max_value;
	}

	QPointF np(px, py);

	_rect = QRectF(np, QPointF(px2, py2));
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

	if (_sel_mode == HORIZONTAL) {
		nrl = _x_min_value;
		nrr = _x_max_value;
	} else if (_sel_mode == VERTICAL) {
		nrt = _y_min_value;
		nrb = _y_max_value;
	}

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
 * PVParallelView::PVSelectionRectangleItem::set_selection_mode
 *****************************************************************************/

void PVParallelView::PVSelectionRectangleItem::set_selection_mode(int sel_mode)
{
	SelectionMode smode = (SelectionMode)sel_mode;

	if (smode == _sel_mode) {
		return;
	}

	_sel_mode = (SelectionMode)sel_mode;

	/* it has been decided that changing the mode implies clearing the
	 * current rectanhle
	 */
	clear();

	std::cout << "PVSelectionRectangleItem::set_selection_mode(" << sel_mode << ")" <<std::endl;
	switch(sel_mode) {
	case RECTANGLE:
		for(auto it : _handles) {
			it->force_hidden(false);
		}
		break;
	case HORIZONTAL:
		for(auto it : _handles) {
			if (it->is_type(PVSelectionHandleItem::W | PVSelectionHandleItem::E)) {
				it->force_hidden(true);
			} else {
				it->force_hidden(false);
			}
		}
		break;
	case VERTICAL:
		for(auto it : _handles) {
			if (it->is_type(PVSelectionHandleItem::N | PVSelectionHandleItem::S)) {
				it->force_hidden(true);
			} else {
				it->force_hidden(false);
			}
		}
		break;
	}
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

void PVParallelView::PVSelectionRectangleItem::set_rect(const QRectF& rect,
                                                        bool commit)
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

	if (commit) {
		emit geometry_has_changed(old_rect, _rect);
	}

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
			h->setZValue(2. * zvalue);
		}
		_central_handle->setZValue(3. * zvalue);
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
