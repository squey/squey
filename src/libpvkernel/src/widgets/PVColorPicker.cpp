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

#include <pvkernel/core/PVAlgorithms.h>
#include <pvkernel/widgets/PVColorPicker.h>
#include <qbrush.h>
#include <qcolor.h>
#include <qflags.h>
#include <qlist.h>
#include <qnamespace.h>
#include <qpoint.h>
#include <qpolygon.h>
#include <qrect.h>
#include <qsize.h>
#include <qtmetamacros.h>
#include <qwidget.h>
#include <stdint.h>
#include <QMouseEvent>
#include <QPainter>
#include <utility>

#include "pvkernel/core/PVHSVColor.h"

#define CROSS_HEIGHT 20
#define CROSS_WIDTH 20
#define CROSS_THICK 3
#define HEIGHT_TRIANGLE 6
#define WIDTH_TRIANGLE 13
#define VERT_MARGIN 5
#define HORI_MARGIN VERT_MARGIN

/*****************************************************************************
 * PVWidgets::PVColorPicker::PVColorPicker
 *****************************************************************************/

PVWidgets::PVColorPicker::PVColorPicker(QWidget* parent)
    : QWidget(parent)
    , _x0(0)
    , _x1(PVCore::PVHSVColor::color_max - 1)
    , _c(_x0)
    , _c1(_x1)
    , _mode(SelectionSingle)
{
	setFocusPolicy(Qt::StrongFocus);

	setContentsMargins(HORI_MARGIN, VERT_MARGIN, HORI_MARGIN, VERT_MARGIN);
}

/*****************************************************************************
 * PVWidgets::PVColorPicker::PVColorPicker
 *****************************************************************************/

PVWidgets::PVColorPicker::PVColorPicker(PVCore::PVHSVColor const& c, QWidget* parent)
    : PVColorPicker(parent)
{
	set_color(c);
}

/*****************************************************************************
 * PVWidgets::PVColorPicker::set_color
 *****************************************************************************/

void PVWidgets::PVColorPicker::set_color(PVCore::PVHSVColor const& c)
{
	if (!is_interval_mode()) {
		if (c == HSV_COLOR_WHITE) {
			// Special case for white color
			_c = c;
		} else {
			_c.h() = PVCore::clamp(c.h(), x0(), x1());
		}
		update();
	}
}

/*****************************************************************************
 * PVWidgets::PVColorPicker::set_interval
 *****************************************************************************/

void PVWidgets::PVColorPicker::set_interval(PVCore::PVHSVColor const& c0,
                                            PVCore::PVHSVColor const& c1)
{
	if (is_interval_mode()) {
		_c.h() = PVCore::clamp(c0.h(), x0(), x1());
		_c1.h() = PVCore::clamp(c1.h(), x0(), x1());
		if (_c.h() > _c1.h()) {
			std::swap(_c, _c1);
		}
		update();
	}
}

/*****************************************************************************
 * PVWidgets::PVColorPicker::screen_x_to_h
 *****************************************************************************/

uint8_t PVWidgets::PVColorPicker::screen_x_to_h(int x) const
{
	x = PVCore::clamp(x, 0, size().width());
	const int width = contentsRect().width() - 1;
	uint8_t h = (uint8_t)((((x - contentsRect().left()) * x_interval()) / width) + x0());
	return h;
}

/*****************************************************************************
 * PVWidgets::PVColorPicker::h_to_screen_x
 *****************************************************************************/

int PVWidgets::PVColorPicker::h_to_screen_x(uint8_t h) const
{
	h = PVCore::clamp(h, x0(), x1());
	const int width = contentsRect().width() - 1;
	const int view_h = h;
	int ret = ((view_h - x0()) * width) / (x_interval());
	return ret + contentsRect().left();
}

/*****************************************************************************
 * PVWidgets::PVColorPicker::sizeHint
 *****************************************************************************/

QSize PVWidgets::PVColorPicker::sizeHint() const
{
	return {PVCore::PVHSVColor::color_max, 10};
}

/*****************************************************************************
 * PVWidgets::PVColorPicker::mousePressEvent
 *****************************************************************************/

void PVWidgets::PVColorPicker::mousePressEvent(QMouseEvent* event)
{
	process_mouse_event(event);
}

/*****************************************************************************
 * PVWidgets::PVColorPicker::mouseMoveEvent
 *****************************************************************************/

void PVWidgets::PVColorPicker::mouseMoveEvent(QMouseEvent* event)
{
	process_mouse_event(event);
}

/*****************************************************************************
 * PVWidgets::PVColorPicker::resizeEvent
 *****************************************************************************/

void PVWidgets::PVColorPicker::resizeEvent(QResizeEvent* /*event*/)
{
}

/*****************************************************************************
 * PVWidgets::PVColorPicker::process_mouse_event
 *****************************************************************************/

void PVWidgets::PVColorPicker::process_mouse_event(QMouseEvent* event)
{
	uint8_t h = screen_x_to_h(event->position().toPoint().x());
	if (event->buttons() == Qt::LeftButton) {
		if (h > _c1.h()) {
			return;
		}
		update_h_left(h);
		update();
	}

	if (is_interval_mode() && (event->buttons() == Qt::RightButton)) {
		if (h < _c.h()) {
			return;
		}
		update_h_right(h);
		update();
	}
}

/*****************************************************************************
 * PVWidgets::PVColorPicker::paintEvent
 *****************************************************************************/

void PVWidgets::PVColorPicker::paintEvent(QPaintEvent* /*event*/)
{
	QPainter painter(this);
	painter.setRenderHint(QPainter::SmoothPixmapTransform, true);
	const int c0_x = h_to_screen_x(_c.h());
	const int c1_x = h_to_screen_x(_c1.h());

	const int height = size().height();
	QRect const& draw_rect = contentsRect();

	for (int x = draw_rect.left(); x <= draw_rect.right(); ++x) {
		const uint8_t h = screen_x_to_h(x);
		QColor color = PVCore::PVHSVColor(h).toQColor();
		if (is_interval_mode()) {
			if ((x < c0_x) || (x > c1_x)) {
				color = color.darker(200);
			}
		}
		painter.fillRect(QRect(x, VERT_MARGIN, 1, height - 2 * VERT_MARGIN), color);
	}

	// For the white color, do not draw anything
	if (_c != HSV_COLOR_WHITE || is_interval_mode()) {
		draw_up_triangle(c0_x, painter);
		draw_down_triangle(c0_x, painter);
		if (is_interval_mode()) {
			draw_up_triangle(c1_x, painter);
			draw_down_triangle(c1_x, painter);
		}
	}
}

/*****************************************************************************
 * PVWidgets::PVColorPicker::draw_up_triangle
 *****************************************************************************/

void PVWidgets::PVColorPicker::draw_up_triangle(int x, QPainter& painter)
{
	QPolygon triangle;
	triangle << QPoint(x - (WIDTH_TRIANGLE - 1) / 2, 1) << QPoint(x + (WIDTH_TRIANGLE - 1) / 2, 1)
	         << QPoint(x, HEIGHT_TRIANGLE + 1);
	painter.setBrush(Qt::SolidPattern);
	painter.drawConvexPolygon(triangle);
}

/*****************************************************************************
 * PVWidgets::PVColorPicker::draw_down_triangle
 *****************************************************************************/

void PVWidgets::PVColorPicker::draw_down_triangle(int x, QPainter& painter)
{
	const int height = size().height() - 2;
	QPolygon triangle;
	triangle << QPoint(x - (WIDTH_TRIANGLE - 1) / 2, height)
	         << QPoint(x + (WIDTH_TRIANGLE - 1) / 2, height) << QPoint(x, height - HEIGHT_TRIANGLE);
	painter.setBrush(QBrush(Qt::white));
	painter.drawConvexPolygon(triangle);
}

/*****************************************************************************
 * PVWidgets::PVColorPicker::update_h_left
 *****************************************************************************/

void PVWidgets::PVColorPicker::update_h_left(uint8_t h)
{
	uint8_t h_max = _c1.h();
	if (!allow_empty_interval()) {
		h_max--;
	}
	_c = PVCore::PVHSVColor(PVCore::clamp(h, x0(), h_max));
	Q_EMIT color_changed_left(_c.h());
}

/*****************************************************************************
 * PVWidgets::PVColorPicker::update_h_right
 *****************************************************************************/

void PVWidgets::PVColorPicker::update_h_right(uint8_t h)
{
	uint8_t h_min = _c.h();
	if (!allow_empty_interval()) {
		h_min++;
	}
	_c1 = PVCore::PVHSVColor(PVCore::clamp(h, h_min, x1()));
	Q_EMIT color_changed_right(_c1.h());
}
