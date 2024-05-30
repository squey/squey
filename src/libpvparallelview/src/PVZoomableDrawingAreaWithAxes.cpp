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

#include <pvkernel/core/PVLogger.h>
#include <pvkernel/core/PVAlgorithms.h>

#include <squey/PVScaled.h>

#include <pvparallelview/PVZoomableDrawingAreaWithAxes.h>
#include <pvparallelview/common.h>

#include <QString>
#include <QPainter>

#include <iostream>

#define SMALL_TICK_LENGTH 3
#define TICK_LENGTH (SMALL_TICK_LENGTH * 2)
#define SCALE_VALUE_OFFSET 8

#define SUBTICK_RATIO 0.45

#define DEFAULT_HMARGIN 50
#define DEFAULT_VMARGIN 50

#define AXIS_MARGIN 0

#define print_r(R) print_rect(R)
#define print_rect(R) __print_rect(#R, R)

template <typename R>
void __print_rect(const char* text, const R& r)
{
	std::cout << text << ": " << r.x() << " " << r.y() << ", " << r.width() << " " << r.height()
	          << std::endl;
}

#define print_s(V) print_scalar(V)
#define print_scalar(V) __print_scalar(#V, V)

template <typename V>
void __print_scalar(const char* text, const V& v)
{
	std::cout << text << ": " << v << std::endl;
}

/**
 * NOTE:
 * * do not forget the scene is defined in (0, -N, N, N), the screen's top value
 *   is also *negative*.
 */

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaWithAxes::PVZoomableDrawingAreaWithAxes
 *****************************************************************************/

PVParallelView::PVZoomableDrawingAreaWithAxes::PVZoomableDrawingAreaWithAxes(QWidget* parent)
    : PVZoomableDrawingArea(parent), _ticks_per_level(10)
{
	set_scene_margins(DEFAULT_HMARGIN, DEFAULT_HMARGIN, DEFAULT_VMARGIN, DEFAULT_VMARGIN);
	set_alignment(Qt::AlignLeft | Qt::AlignBottom);
	set_transformation_anchor(PVWidgets::PVGraphicsView::AnchorUnderMouse);
	set_horizontal_scrollbar_policy(Qt::ScrollBarAlwaysOn);
	set_vertical_scrollbar_policy(Qt::ScrollBarAlwaysOn);
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaWithAxes::~PVZoomableDrawingAreaWithAxes
 *****************************************************************************/

PVParallelView::PVZoomableDrawingAreaWithAxes::~PVZoomableDrawingAreaWithAxes() = default;

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaWithAxes::set_decoration_color
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingAreaWithAxes::set_decoration_color(const QColor& color)
{
	if (_decoration_color != color) {
		_decoration_color = color;
		get_viewport()->update();
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaWithAxes::set_x_legend
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingAreaWithAxes::set_x_legend(const QString& legend)
{
	if (_x_legend != legend) {
		_x_legend = legend;
		get_viewport()->update();
	}
}

void PVParallelView::PVZoomableDrawingAreaWithAxes::set_x_legend(PVWidgets::PVAxisComboBox* legend)
{
	if (_x_legend_widget != legend) {
		_x_legend_widget = legend;
		_x_legend_widget->setParent(this);
		_x_legend_widget->adjustSize();
		_x_legend_widget->move(get_viewport()->rect().width() - _x_legend_widget->width(),
		                       get_viewport()->rect().height() - _x_legend_widget->height());
		get_viewport()->update();
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaWithAxes::set_y_legend
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingAreaWithAxes::set_y_legend(const QString& legend)
{
	if (_y_legend != legend) {
		_y_legend = legend;
		get_viewport()->update();
	}
}

void PVParallelView::PVZoomableDrawingAreaWithAxes::set_y_legend(PVWidgets::PVAxisComboBox* legend)
{
	if (_y_legend_widget != legend) {
		_y_legend_widget = legend;
		_y_legend_widget->setParent(this);
		_y_legend_widget->adjustSize();
		_y_legend_widget->move(0, 0);
		get_viewport()->update();
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaWithAxes::set_ticks_count
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingAreaWithAxes::set_ticks_per_level(int n)
{
	if (_ticks_per_level != n) {
		_ticks_per_level = n;
		get_viewport()->update();
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaWithAxes::get_x_value_at
 *****************************************************************************/

QString PVParallelView::PVZoomableDrawingAreaWithAxes::get_x_value_at(const qint64 pos)
{
	return get_elided_text(QString::number(pos));
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaWithAxes::get_y_value_at
 *****************************************************************************/

QString PVParallelView::PVZoomableDrawingAreaWithAxes::get_y_value_at(const qint64 pos)
{
	return get_elided_text(QString::number(pos));
}

void PVParallelView::PVZoomableDrawingAreaWithAxes::recompute_margins()
{
	recompute_decorations();
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaWithAxes::recompute_decorations
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingAreaWithAxes::recompute_decorations()
{
	QFontMetrics fm(get_viewport()->font());
	// QSize viewport_rect = get_viewport()->size();

	int margin_top = get_scene_top_margin();
	int margin_left = get_scene_left_margin();
	int margin_right = get_scene_right_margin();
	int margin_bottom = get_scene_bottom_margin();
	qreal right = get_viewport()->width() - margin_right;
	qreal bottom = get_viewport()->height() - margin_bottom;

	QRectF view_in_scene = get_visible_scene_rect();
	QRectF scene_in_screen = map_from_scene(get_scene_rect());

	_x_axis_length = std::min(scene_in_screen.width(), right - margin_left - 1.);
	_y_axis_length = std::min(scene_in_screen.height(), bottom - DEFAULT_HMARGIN);
	int t = bottom - _y_axis_length;

	int l = std::max(
	    fm.boundingRect(get_y_value_at(-view_in_scene.y())).width(),
	    fm.boundingRect(get_y_value_at(-(view_in_scene.y() + view_in_scene.height()))).width());

	int r = fm.boundingRect(get_x_value_at(-(view_in_scene.x() + view_in_scene.width()))).width();

	l += 2 * SCALE_VALUE_OFFSET;
	r = (r / 2) + SCALE_VALUE_OFFSET;

	l = std::max(l, DEFAULT_HMARGIN);
	r = std::max(r, DEFAULT_HMARGIN);

	if ((l != margin_left) || (r != margin_right) || (t != margin_top)) {
		set_scene_margins(l, r, t, margin_bottom);
	}

	if (_x_legend_widget) {
		_x_legend_widget->adjustSize();
		_x_legend_widget->move(get_viewport()->rect().width() - _x_legend_widget->width(),
		                       get_viewport()->rect().height() - _x_legend_widget->height());
	}
	if (_y_legend_widget) {
		_y_legend_widget->adjustSize();
		_y_legend_widget->move(0, 0);
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaWithAxes::draw_decorations
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingAreaWithAxes::draw_decorations(QPainter* painter,
                                                                     const QRectF& margined_rect)
{
	if (get_scene_rect().isNull()) {
		PVLOG_WARN("using a PVZoomableDrawingArea with no defined scene\n");
		return;
	}

	// draw_deco_v* functions expect a painter that uses the viewport coordinate
	// system.
	QRectF viewport_rect = map_from_margined(margined_rect);
	painter->save();
	painter->resetTransform();
	draw_deco_v3(painter, viewport_rect);
	painter->restore();
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaWithAxes::drawBackground
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingAreaWithAxes::drawBackground(QPainter* painter,
                                                                   const QRectF& rect)
{
	draw_decorations(painter, rect);
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaWithAxes::draw_deco_v3
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingAreaWithAxes::draw_deco_v3(QPainter* painter,
                                                                 const QRectF& /*rect*/)
{
	int ticks_per_level = get_ticks_per_level();
	qreal log_ticks_per_level = log(ticks_per_level);

	int left = get_scene_left_margin();
	int right = left + get_x_axis_length();
	int top = get_scene_top_margin();
	int bottom = top + get_y_axis_length();

	QRect viewport_rect = get_viewport()->rect();
	QRectF rect_in_scene = map_to_scene(QRect(0, 0, 1024, 1024)).intersected(get_scene_rect());

	QFontMetrics fm = painter->fontMetrics();

	painter->save();
	painter->resetTransform();
	painter->setPen(QPen(_decoration_color, 0));

	// scales
	painter->drawLine(left - AXIS_MARGIN, top - AXIS_MARGIN, left - AXIS_MARGIN,
	                  bottom + AXIS_MARGIN);
	painter->drawLine(left - AXIS_MARGIN, bottom + AXIS_MARGIN, right, bottom + AXIS_MARGIN);

	// ticks
	int fm_ascent = fm.ascent();

	qreal x_scale = x_zoom_to_scale(get_x_axis_zoom().get_clamped_value());
	qreal x_level = log(rect_in_scene.width()) / log_ticks_per_level;
	qreal scene_subtick_width = pow(ticks_per_level, (int)x_level) / ticks_per_level;
	qreal screen_subtick_width = scene_subtick_width * x_scale;

	bool need_x_subticks = (screen_subtick_width > subtick_min_gap) && (scene_subtick_width >= 1.0);

	qreal scene_left = map_to_scene(QPoint(left, 0)).x();
	int64_t x_subtick_index = ceil(scene_left / scene_subtick_width);
	qreal scene_pos = x_subtick_index * scene_subtick_width;
	qreal screen_pos = map_from_scene(QPointF(scene_pos, 0)).x();

	if (x_axis_inverted()) {
		scene_subtick_width = -scene_subtick_width;
	}

	// let's draw
	while ((int)screen_pos <= right) {
		if ((x_subtick_index % ticks_per_level) != 0) {
			if (need_x_subticks) {
				painter->drawLine(round(screen_pos), bottom + AXIS_MARGIN, round(screen_pos),
				                  bottom + SMALL_TICK_LENGTH + AXIS_MARGIN);
			}
		} else {
			// QString s = QString::number(scene_pos, 'f', 2);
			QString s = get_x_value_at(scene_pos);
			int s_len = fm.boundingRect(s).width();

			painter->drawText(screen_pos - (s_len / 2), bottom + SCALE_VALUE_OFFSET + fm_ascent, s);
			painter->drawLine(round(screen_pos), bottom + AXIS_MARGIN, round(screen_pos),
			                  bottom + TICK_LENGTH + AXIS_MARGIN);
		}

		screen_pos += screen_subtick_width;
		scene_pos += scene_subtick_width;
		++x_subtick_index;
	}

	qreal y_scale = y_zoom_to_scale(get_y_axis_zoom().get_clamped_value());

	qreal y_level = log(rect_in_scene.height()) / log_ticks_per_level;
	qreal scene_subtick_height = pow(ticks_per_level, (int)y_level) / ticks_per_level;
	qreal screen_subtick_height = scene_subtick_height * y_scale;

	bool need_y_subticks =
	    (screen_subtick_height > subtick_min_gap) && (scene_subtick_height >= 1.0);

	qreal scene_top = map_to_scene(QPoint(0, top)).y();
	int64_t y_subtick_index = ceil(scene_top / scene_subtick_height);

	scene_pos = y_subtick_index * scene_subtick_height;
	screen_pos = map_from_scene(QPointF(0, scene_pos)).y();

	if (y_axis_inverted()) {
		scene_subtick_height = -scene_subtick_height;
	}

	// let's draw
	while ((int)screen_pos <= bottom) {
		if ((y_subtick_index % ticks_per_level) != 0) {
			if (need_y_subticks) {
				painter->drawLine(left - AXIS_MARGIN, round(screen_pos),
				                  left - SMALL_TICK_LENGTH - AXIS_MARGIN, round(screen_pos));
			}
		} else {
			// QString s = QString::number(-scene_pos, 'f', 2);
			QString s = get_y_value_at(scene_pos);
			int s_len = fm.boundingRect(s).width();

			painter->drawText(left - AXIS_MARGIN - s_len - SCALE_VALUE_OFFSET, round(screen_pos),
			                  s);

			painter->drawLine(left - AXIS_MARGIN, round(screen_pos),
			                  left - TICK_LENGTH - AXIS_MARGIN, round(screen_pos));
		}

		screen_pos += screen_subtick_height;
		scene_pos += scene_subtick_height;
		++y_subtick_index;
	}

	// legends

	QFont f(painter->font());
	f.setWeight(QFont::Bold);
	painter->setFont(f);

	fm = painter->fontMetrics();

	if (!_x_legend.isNull()) {
		const QSize text_size = fm.size(Qt::TextSingleLine, _x_legend);
		const int frame_width = text_size.width() + frame_margins.left() + frame_margins.right();
		const int frame_height = text_size.height() + frame_margins.top() + frame_margins.bottom();
		const QRect frame(viewport_rect.width() - frame_width - frame_offsets.right(),
		                  viewport_rect.height() - frame_height - frame_offsets.bottom(),
		                  frame_width, frame_height);

		painter->setPen(Qt::NoPen);

		painter->drawRect(frame);

		painter->setPen(QPen(frame_text_color, 0));
		painter->setBrush(Qt::NoBrush);

		painter->drawText(frame.left() + frame_margins.left(),
		                  frame.top() + frame_margins.top() + fm.ascent(), _x_legend);
	}

	if (!_y_legend.isNull()) {
		const QSize text_size = fm.size(Qt::TextSingleLine, _y_legend);
		const QRect frame(frame_offsets.left(), frame_offsets.top(),
		                  text_size.width() + frame_margins.left() + frame_margins.right(),
		                  text_size.height() + frame_margins.top() + frame_margins.bottom());

		painter->setPen(Qt::NoPen);

		painter->drawRect(frame);

		painter->setPen(QPen(frame_text_color, 0));
		painter->setBrush(Qt::NoBrush);

		painter->drawText(frame.left() + frame_margins.left(),
		                  frame.top() + frame_margins.top() + fm.ascent(), _y_legend);
	}

	painter->restore();
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaWithAxes::get_elided_text
 *****************************************************************************/
QString PVParallelView::PVZoomableDrawingAreaWithAxes::get_elided_text(const QString& text) const
{
	QFontMetrics fm(get_viewport()->font());

	// MAX_TEXT_LABEL_WIDTH value should be calculated, depend of the client's windows settings.
	if (fm.horizontalAdvance(text) > MAX_TEXT_LABEL_WIDTH) {
		return fm.elidedText(text, Qt::ElideMiddle, MAX_TEXT_LABEL_WIDTH);
	}
	return text;
}
