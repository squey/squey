/**
 * \file PVColorPicker.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <pvkernel/core/PVAlgorithms.h>
#include <pvkernel/widgets/PVColorPicker.h>

#include <QMouseEvent>
#include <QPainter>
#include <QPaintEvent>

#include <iostream>

#define CROSS_HEIGHT 20
#define CROSS_WIDTH 20
#define CROSS_THICK 3
#define HEIGHT_TRIANGLE 8
#define WIDTH_TRIANGLE 15
#define VERT_MARGIN 4
#define WHITE_COLOR_RANGE 3

PVWidgets::PVColorPicker::PVColorPicker(QWidget* parent):
	QFrame(parent)
{
	init();
}

PVWidgets::PVColorPicker::PVColorPicker(PVCore::PVHSVColor const& c, QWidget* parent):
	QFrame(parent)
{
	init();
	set_color(c);
}

void PVWidgets::PVColorPicker::init()
{
	setFocusPolicy(Qt::StrongFocus);

	_x0 = 0;
	_x1 = PVCore::PVHSVColor::color_max;
	_c.h()  = _x0;
	_c1.h() = _x1;
	_mode = SelectionSingle;
	setFocusPolicy(Qt::StrongFocus);
	setFrameShape(QFrame::Box);
	setFrameShadow(QFrame::Sunken);
	setLineWidth(1);
}

void PVWidgets::PVColorPicker::set_color(PVCore::PVHSVColor const& c)
{
	if (!is_interval_mode()) {
		if (c.h() == HSV_COLOR_WHITE) {
			// Special case for white color
			_c = c;
		}
		else {
			_c.h() = PVCore::clamp(c.h(), x0(), x1());
		}
		update();
	}
}

void PVWidgets::PVColorPicker::set_interval(PVCore::PVHSVColor const& c0, PVCore::PVHSVColor const& c1)
{
	if (is_interval_mode()) {
		_c.h()  = PVCore::clamp(c0.h(), x0(), x1());
		_c1.h() = PVCore::clamp(c1.h(), x0(), x1());
		if (_c.h() > _c1.h()) {
			std::swap(_c, _c1);
		}
		update();
	}
}

uint8_t PVWidgets::PVColorPicker::screen_x_to_h(const int x) const
{
	const int width  = size().width();
	uint8_t h = (uint8_t) (((x*x_interval())/width) + x0());
	return h;
}

int PVWidgets::PVColorPicker::h_to_x_screen(uint8_t h) const
{
	h = PVCore::clamp(h, x0(), x1());
	const int width  = size().width();
	const int view_h = h;
	int ret = ((view_h-x0())*width)/(x_interval());
	return ret;
}

QSize PVWidgets::PVColorPicker::sizeHint() const
{
	return QSize(HSV_COLOR_COUNT, 10);
}

void PVWidgets::PVColorPicker::mousePressEvent(QMouseEvent* event)
{
	process_mouse_event(event);
}

void PVWidgets::PVColorPicker::mouseMoveEvent(QMouseEvent* event)
{
	process_mouse_event(event);
}

void PVWidgets::PVColorPicker::resizeEvent(QResizeEvent* /*event*/)
{
}

void PVWidgets::PVColorPicker::process_mouse_event(QMouseEvent* event)
{
	uint8_t h = screen_x_to_h(event->x());
	if ((event->buttons() & Qt::LeftButton) == Qt::LeftButton) {
		if (h > _c1.h()) {
			return;
		}
		update_h_left(h);
		update();
	}

	if (is_interval_mode() && 
	    ((event->buttons() & Qt::RightButton) == Qt::RightButton)) {
		if (h < _c.h()) {
			return;
		}
		update_h_right(h);
		update();
	}
}

void PVWidgets::PVColorPicker::paintEvent(QPaintEvent* event)
{
	QPainter painter(this);
	painter.setRenderHint(QPainter::SmoothPixmapTransform, true);
	QRect const& rect = event->rect();
	const int height = size().height();

	QRect draw_rect = rect.intersect(contentsRect());

	const int c0_x = h_to_x_screen(_c.h());
	const int c1_x = h_to_x_screen(_c1.h());

	QColor color;
	int x = draw_rect.left();
	for (; x < draw_rect.right() - WHITE_COLOR_RANGE; x++) {
		int real_x = x-contentsRect().x();
		const uint8_t h = screen_x_to_h(real_x);
		PVCore::PVHSVColor(h).toQColor(color);
		if (is_interval_mode()) {
			if (h < _c.h() || h >= _c1.h()) {
				color = color.darker(200);
			}
		}
		painter.fillRect(QRect(x, VERT_MARGIN, 1, height-2*VERT_MARGIN), color);
	}
	for (; x < draw_rect.right(); x++) {
		painter.fillRect(QRect(x, VERT_MARGIN, 1, height-2*VERT_MARGIN), QColor(Qt::white));
	}


	// For the white color, do not draw anything
	if (_c.h() != HSV_COLOR_WHITE || is_interval_mode()) {
		draw_up_triangle(c0_x, painter);
		draw_down_triangle(c0_x, painter);
		if (is_interval_mode()) {
			draw_up_triangle(c1_x, painter);
			draw_down_triangle(c1_x, painter);
		}
	}

	// Draw the cross
	//painter.fillRect(QRect(_cross.x() - CROSS_WIDTH/2, _cross.y() - CROSS_THICK/2, CROSS_WIDTH, CROSS_THICK), Qt::black);
	//painter.fillRect(QRect(_cross.x() - CROSS_THICK/2, _cross.y() - CROSS_WIDTH/2, CROSS_THICK, CROSS_WIDTH), Qt::black);

	QFrame::paintEvent(event);
}

void PVWidgets::PVColorPicker::draw_up_triangle(int x, QPainter& painter)
{
	QPolygon triangle;
	triangle << QPoint(x - (WIDTH_TRIANGLE-1)/2, 0) << QPoint(x + (WIDTH_TRIANGLE-1)/2, 0)
	         << QPoint(x, HEIGHT_TRIANGLE);
	painter.setBrush(Qt::SolidPattern);
	painter.drawConvexPolygon(triangle);
}

void PVWidgets::PVColorPicker::draw_down_triangle(int x, QPainter& painter)
{
	const int height = size().height();
	QPolygon triangle;
	triangle << QPoint(x - (WIDTH_TRIANGLE-1)/2, height) << QPoint(x + (WIDTH_TRIANGLE-1)/2, height)
	         << QPoint(x, height-HEIGHT_TRIANGLE);
	painter.setBrush(QBrush(Qt::white));
	painter.drawConvexPolygon(triangle);
}

void PVWidgets::PVColorPicker::update_h_left(uint8_t h)
{
	uint8_t h_max = _c1.h();
	if (!allow_empty_interval()) {
		h_max--;
	}
	_c = PVCore::clamp(h, x0(), h_max);
	if (_c.h() > PVCore::PVHSVColor::color_max - WHITE_COLOR_RANGE -2) {
		_c.h() = HSV_COLOR_WHITE;
	}

	emit color_changed_left(_c.h());
}

void PVWidgets::PVColorPicker::update_h_right(uint8_t h)
{
	uint8_t h_min = _c.h();
	if (!allow_empty_interval()) {
		h_min++;
	}
	_c1 = PVCore::clamp(h, h_min, x1());
	emit color_changed_right(_c1.h());
}
