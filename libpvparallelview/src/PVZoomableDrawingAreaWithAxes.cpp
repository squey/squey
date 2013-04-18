
#include <pvkernel/core/PVAlgorithms.h>

#include <pvparallelview/PVZoomableDrawingAreaWithAxes.h>

#include <QString>
#include <QPainter>

#include <iostream>

#define SMALL_TICK_LENGTH 3
#define TICK_LENGTH (SMALL_TICK_LENGTH * 2)
#define SCALE_VALUE_OFFSET 8

#define SUBTICK_RATIO 0.45
#define SUBTICK_MINSIZE 32

#define DEFAULT_HMARGIN 50
#define DEFAULT_VMARGIN 40

#define print_r(R) print_rect(R)
#define print_rect(R) __print_rect(#R, R)

template <typename R>
void __print_rect(const char *text, const R &r)
{
	std::cout << text << ": "
	          << r.x() << " " << r.y() << ", "
	          << r.width() << " " << r.height()
	          << std::endl;
}

#define print_s(V) print_scalar(V)
#define print_scalar(V) __print_scalar(#V, V)

template <typename V>
void __print_scalar(const char *text, const V &v)
{
	std::cout << text << ": "
	          << v
	          << std::endl;
}

/**
 * NOTE:
 * * do not forget the scene is defined in (0, -N, N, N), the screen's top value
 *   is also *negative*.
 */

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaWithAxes::PVZoomableDrawingAreaWithAxes
 *****************************************************************************/

PVParallelView::PVZoomableDrawingAreaWithAxes::PVZoomableDrawingAreaWithAxes(QWidget *parent) :
	PVZoomableDrawingArea(parent),
	_ticks_per_level(10),
	_first_resize(true)
{
	set_scene_margins(DEFAULT_HMARGIN, DEFAULT_HMARGIN,
	                  DEFAULT_VMARGIN, DEFAULT_VMARGIN);
	set_alignment(Qt::AlignLeft | Qt::AlignBottom);
	set_transformation_anchor(PVWidgets::PVGraphicsView::AnchorUnderMouse);
	set_horizontal_scrollbar_policy(Qt::ScrollBarAlwaysOn);
	set_vertical_scrollbar_policy(Qt::ScrollBarAlwaysOn);
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaWithAxes::~PVZoomableDrawingAreaWithAxes
 *****************************************************************************/

PVParallelView::PVZoomableDrawingAreaWithAxes::~PVZoomableDrawingAreaWithAxes()
{}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaWithAxes::set_decoration_color
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingAreaWithAxes::set_decoration_color(const QColor &color)
{
	if (_decoration_color != color) {
		_decoration_color = color;
		update();
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaWithAxes::set_x_legend
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingAreaWithAxes::set_x_legend(const QString &legend)
{
	if (_x_legend != legend) {
		_x_legend = legend;
		update();
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaWithAxes::set_y_legend
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingAreaWithAxes::set_y_legend(const QString &legend)
{
	if (_y_legend != legend) {
		_y_legend = legend;
		update();
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaWithAxes::set_ticks_count
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingAreaWithAxes::set_ticks_per_level(int n)
{
	if (_ticks_per_level != n) {
		_ticks_per_level = n;
		update();
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaWithAxes::get_x_value_at
 *****************************************************************************/

QString PVParallelView::PVZoomableDrawingAreaWithAxes::get_x_value_at(const qint64 pos) const
{
	return QString::number(pos);
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaWithAxes::get_y_value_at
 *****************************************************************************/

QString PVParallelView::PVZoomableDrawingAreaWithAxes::get_y_value_at(const qint64 pos) const
{
	return QString::number(pos);
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaWithAxes::recompute_decorations
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingAreaWithAxes::recompute_decorations(QPainter *painter,
                                                                          const QRectF &rect)
{
	int margin_top = get_scene_top_margin();
	int margin_left = get_scene_left_margin();
	int margin_right = get_scene_right_margin();
	int margin_bottom = get_scene_bottom_margin();
	qreal right = get_viewport()->width() - margin_right;
	qreal bottom = get_viewport()->height() - margin_bottom;

	QRectF view_in_scene = map_to_scene(QRectF(margin_left,
	                                           margin_top,
	                                           rect.width() - margin_right,
	                                           rect.height() - margin_bottom));

	QRectF scene_in_screen = map_from_scene(get_scene_rect());

	_x_axis_length = PVCore::min(scene_in_screen.width(), right - margin_left - 1.);
	_y_axis_length = PVCore::min(scene_in_screen.height(), bottom - DEFAULT_HMARGIN);
	int t = bottom - _y_axis_length;

	int l = PVCore::max(painter->fontMetrics().boundingRect(get_y_value_at(-view_in_scene.y())).width(),
	                    painter->fontMetrics().boundingRect(get_y_value_at(-(view_in_scene.y() + view_in_scene.height()))).width());
	int r = painter->fontMetrics().boundingRect(get_y_value_at(-(view_in_scene.x() + view_in_scene.width()))).width();

	l += 2 * SCALE_VALUE_OFFSET;
	r = (r / 2) + SCALE_VALUE_OFFSET;

	if ((l > margin_left) || ( r > margin_right) || (t != margin_top)) {
		set_scene_margins(l, r, t, margin_bottom);
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaWithAxes::draw_decorations
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingAreaWithAxes::draw_decorations(QPainter *painter,
                                                                     const QRectF &rect)
{
	draw_deco_v3(painter, rect);
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaWithAxes::drawBackground
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingAreaWithAxes::drawBackground(QPainter *painter,
                                                                   const QRectF &rect)
{
	recompute_decorations(painter, rect);
	draw_decorations(painter, rect);
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaWithAxes::resizeEvent
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingAreaWithAxes::resizeEvent(QResizeEvent *event)
{
	/**
	 * as the first resize event happens only at the first call to
	 * QWidget::show(), the viewport can not also be correctly set before
	 * it happens. So that the ::center_on() must be at the first resize...
	 */
	PVParallelView::PVZoomableDrawingArea::resizeEvent(event);
	if (_first_resize) {
		center_on(0., 0.);
		_first_resize = false;
	}
}


void PVParallelView::PVZoomableDrawingAreaWithAxes::draw_deco_v1(QPainter *painter,
                                                                 const QRectF &rect)
{
	qreal top = get_scene_top_margin();
	qreal left = get_scene_left_margin();
	//qreal right = rect.width() - get_scene_right_margin();
	qreal margin_bottom = get_scene_bottom_margin();
	qreal bottom = rect.height() - margin_bottom;

	painter->save();
	painter->resetTransform();
	painter->setPen(_decoration_color);

	QFontMetrics fm = painter->fontMetrics();
	int fm_ascent = fm.ascent();

	/* x scale
	 */

	// axis
	painter->drawLine(left, bottom - _y_axis_length, left, bottom);

	// legend
	if (!_x_legend.isNull()) {
		painter->drawText(left, bottom,
		                  _x_axis_length, margin_bottom,
		                  Qt::AlignRight | Qt::AlignBottom,
		                  _x_legend);
	}

	// ticks
	qreal x_step = _x_axis_length / (qreal)_ticks_per_level;
	for(int i = 0; i <= _ticks_per_level; ++i) {
		qreal v = i * x_step;
		QString s = get_x_value_at(map_to_scene(left + i * x_step, 0).x());
		int s_len = fm.boundingRect(s).width();
		painter->drawLine(left + v, bottom, left + v, bottom + TICK_LENGTH);
		painter->drawText(left + v - (s_len / 2),
		                  bottom + SCALE_VALUE_OFFSET + fm_ascent,
		                  s);
	}

	/* y scale
	 */

	// axis
	painter->drawLine(left, bottom, left + _x_axis_length, bottom);

	// legend
	if (!_y_legend.isNull()) {
		painter->drawText(left, 0,
		                  _x_axis_length, top,
		                  Qt::AlignLeft | Qt::AlignTop,
		                  _y_legend);
	}

	// ticks
	qreal y_step = _y_axis_length / (qreal)_ticks_per_level;
	for(int i = 0; i <= _ticks_per_level; ++i) {
		qreal v = i * y_step;
		QString s = get_y_value_at(-map_to_scene(0, top + i * y_step).y());
		int s_len = fm.boundingRect(s).width();
		painter->drawLine(left, bottom - _y_axis_length + v,
		                  left - TICK_LENGTH, bottom - _y_axis_length + v);
		painter->drawText(left - s_len - SCALE_VALUE_OFFSET,
		                  bottom - _y_axis_length + v,
		                  s);
	}

	painter->restore();
}

void PVParallelView::PVZoomableDrawingAreaWithAxes::draw_deco_v2(QPainter *painter,
                                                                 const QRectF &rect)
{
	QRectF scene_in_screen = map_from_scene(get_scene_rect());
	// FIXME!
	if (scene_in_screen.width() == 0 || scene_in_screen.height() == 0) {
		return;
	}
	int ticks_per_level = get_ticks_per_level();
	qreal log_ticks_per_level = log(ticks_per_level);
	int base_tick_size = 1024 / ticks_per_level;
	// int base_tick_size = pow(ticks_per_level, round(log(1024) / log_ticks_per_level)) / ticks_per_level;

	int left = get_scene_left_margin();
	int right = left + get_x_axis_length();
	int top = get_scene_top_margin();
	int margin_bottom = get_y_axis_length();
	int bottom = top + margin_bottom;

	painter->save();
	painter->resetTransform();
	painter->setPen(Qt::white);

	// scales
	painter->drawLine(left, top, left, bottom);
	painter->drawLine(left, bottom, right, bottom);

	// legends
	if (!_x_legend.isNull()) {
		painter->drawText(left, bottom,
		                  _x_axis_length, margin_bottom,
		                  Qt::AlignRight | Qt::AlignBottom,
		                  _x_legend);
	}

	if (!_y_legend.isNull()) {
		painter->drawText(left, 0,
		                  _x_axis_length, top,
		                  Qt::AlignLeft | Qt::AlignTop,
		                  _y_legend);
	}

	QFontMetrics fm = painter->fontMetrics();
	int fm_ascent = fm.ascent();

	/* horizontal ticks
	 */

	// first we need the ticks subdivision level
	qreal scene_width = scene_in_screen.width();
	qreal x_tick_count = scene_width / base_tick_size;
	// qreal x_tick_count = pow(ticks_per_level, (int)(log(scene_width) / log_ticks_per_level)) / base_tick_size;
	int x_tick_level = (int)(log(x_tick_count) / log_ticks_per_level);
	int64_t x_real_tick_count = pow(ticks_per_level, x_tick_level);

	// to know if intermediate ticks have to be drawn or not
	bool need_x_subticks = (x_real_tick_count / x_tick_count) < SUBTICK_RATIO;

	// and to deduce needed ticks width in screen and scene spaces
	x_real_tick_count *= ticks_per_level;
	qreal screen_subtick_width = scene_width / (x_real_tick_count);
	qreal scene_subtick_width = get_scene_rect().width() / (x_real_tick_count);

	// next, some information about the first tick position
	qreal scene_left = map_to_scene(QPoint(left, 0)).x();
	int64_t x_tick_index = ceil(scene_left / scene_subtick_width);
	qreal t_left = x_tick_index * scene_subtick_width;
	int screen_left = map_from_scene(QPointF(t_left, 0)).x();

	qreal screen_pos = screen_left;
	qreal scene_pos = t_left;

	// let's draw
	while ((int)screen_pos <= right) {
		if ((x_tick_index % ticks_per_level) != 0) {
			if (need_x_subticks) {
				painter->drawLine(round(screen_pos),
				                  bottom,
				                  round(screen_pos),
				                  bottom + SMALL_TICK_LENGTH);
			}
		} else {
			// QString s = QString::number(scene_pos, 'f');
			QString s = get_x_value_at(scene_pos);
			int s_len = fm.boundingRect(s).width();

			painter->drawText(screen_pos - (s_len / 2),
			                  bottom + SCALE_VALUE_OFFSET + fm_ascent,
			                  s);
			painter->drawLine(round(screen_pos),
			                  bottom,
			                  round(screen_pos),
			                  bottom + TICK_LENGTH);
		}

		screen_pos += screen_subtick_width;
		scene_pos += scene_subtick_width;
		++x_tick_index;
	}

	/* vertical scale
	 */

	// first we need the ticks subdivision level
	qreal scene_height = scene_in_screen.height();
	qreal y_tick_count = scene_height / base_tick_size;
	int y_tick_level = (int)(log(y_tick_count) / log_ticks_per_level);
	int64_t y_real_tick_count = pow(ticks_per_level, y_tick_level);

	// to know if intermediate ticks have to be drawn or not
	bool need_y_subticks = (y_real_tick_count / y_tick_count) < SUBTICK_RATIO;

	// and to deduce needed ticks width in screen and scene spaces
	y_real_tick_count *= ticks_per_level;
	qreal screen_subtick_height = scene_height / (y_real_tick_count);
	qreal scene_subtick_height = get_scene_rect().height() / (y_real_tick_count);

	// next, some information about the first tick position
	qreal scene_top = map_to_scene(QPoint(0, top)).y();
	int64_t y_tick_index = ceil(scene_top / scene_subtick_height);
	qreal t_top = y_tick_index * scene_subtick_height;
	qreal screen_top = map_from_scene(QPointF(0, t_top)).y();

	screen_pos = screen_top;
	scene_pos = t_top;

	// let's draw
	while ((int)screen_pos <= bottom) {
		if ((y_tick_index % ticks_per_level) != 0) {
			if (need_y_subticks) {
				painter->drawLine(left,
				                  round(screen_pos),
				                  left - SMALL_TICK_LENGTH,
				                  round(screen_pos));
			}
		} else {
			// QString s = QString::number(-scene_pos, 'f');
			QString s = get_y_value_at(-scene_pos);
			int s_len = fm.boundingRect(s).width();

			painter->drawText(left - s_len - SCALE_VALUE_OFFSET,
			                  round(screen_pos),
			                  s);

			painter->drawLine(left,
			                  round(screen_pos),
			                  left - TICK_LENGTH,
			                  round(screen_pos));

		}

		screen_pos += screen_subtick_height;
		scene_pos += scene_subtick_height;
		++y_tick_index;
	}

	painter->restore();
}

void PVParallelView::PVZoomableDrawingAreaWithAxes::draw_deco_v3(QPainter *painter,
                                                                 const QRectF &rect)
{
	int ticks_per_level = get_ticks_per_level();
	qreal log_ticks_per_level = log(ticks_per_level);

	int left = get_scene_left_margin();
	int right = left + get_x_axis_length();
	int top = get_scene_top_margin();
	int margin_bottom = get_y_axis_length();
	int bottom = top + margin_bottom;

	QRectF rect_in_scene = map_to_scene(QRect(0, 0, 1024, 1024)).intersected(get_scene_rect());

	painter->save();
	painter->resetTransform();
	painter->setPen(_decoration_color);

	// scales
	painter->drawLine(left, top, left, bottom);
	painter->drawLine(left, bottom, right, bottom);

	// legends
	if (!_x_legend.isNull()) {
		painter->drawText(left, bottom,
		                  _x_axis_length, margin_bottom,
		                  Qt::AlignRight | Qt::AlignBottom,
		                  _x_legend);
	}

	if (!_y_legend.isNull()) {
		painter->drawText(left, 0,
		                  _x_axis_length, top,
		                  Qt::AlignLeft | Qt::AlignTop,
		                  _y_legend);
	}

	// ticks
	QFontMetrics fm = painter->fontMetrics();
	int fm_ascent = fm.ascent();

	qreal x_scale = x_zoom_to_scale(get_x_axis_zoom().get_clamped_value());
	qreal x_level = log(rect_in_scene.width()) / log_ticks_per_level;
	qreal scene_subtick_width = pow(ticks_per_level, (int)x_level) / ticks_per_level;
	qreal screen_subtick_width = scene_subtick_width * x_scale;

	bool need_x_subticks = (screen_subtick_width > SUBTICK_MINSIZE);

	qreal scene_left = map_to_scene(QPoint(left, 0)).x();
	int64_t x_subtick_index = ceil(scene_left / scene_subtick_width);
	qreal scene_pos = x_subtick_index * scene_subtick_width;
	qreal screen_pos = map_from_scene(QPointF(scene_pos, 0)).x();

	// let's draw
	while ((int)screen_pos <= right) {
		if ((x_subtick_index % ticks_per_level) != 0) {
			if (need_x_subticks) {
				painter->drawLine(round(screen_pos),
				                  bottom,
				                  round(screen_pos),
				                  bottom + SMALL_TICK_LENGTH);
			}
		} else {
			// QString s = QString::number(scene_pos, 'f', 2);
			QString s = get_x_value_at(scene_pos);
			int s_len = fm.boundingRect(s).width();

			painter->drawText(screen_pos - (s_len / 2),
			                  bottom + SCALE_VALUE_OFFSET + fm_ascent,
			                  s);
			painter->drawLine(round(screen_pos),
			                  bottom,
			                  round(screen_pos),
			                  bottom + TICK_LENGTH);
		}

		screen_pos += screen_subtick_width;
		scene_pos += scene_subtick_width;
		++x_subtick_index;
	}

	qreal y_scale = y_zoom_to_scale(get_y_axis_zoom().get_clamped_value());

	qreal y_level = log(rect_in_scene.height()) / log_ticks_per_level;
	qreal scene_subtick_height = pow(ticks_per_level, (int)y_level) / ticks_per_level;
	qreal screen_subtick_height = scene_subtick_height * y_scale;

	bool need_y_subticks = (screen_subtick_height > SUBTICK_MINSIZE);

	qreal scene_top = map_to_scene(QPoint(0, top)).y();
	int64_t y_subtick_index = ceil(scene_top / scene_subtick_height);

	scene_pos = y_subtick_index * scene_subtick_height;
	screen_pos = map_from_scene(QPointF(0, scene_pos)).y();


	// let's draw
	while ((int)screen_pos <= bottom) {
		if ((y_subtick_index % ticks_per_level) != 0) {
			if (need_y_subticks) {
				painter->drawLine(left,
				                  round(screen_pos),
				                  left - SMALL_TICK_LENGTH,
				                  round(screen_pos));
			}
		} else {
			// QString s = QString::number(-scene_pos, 'f', 2);
			QString s = get_y_value_at(-scene_pos);
			int s_len = fm.boundingRect(s).width();

			painter->drawText(left - s_len - SCALE_VALUE_OFFSET,
			                  round(screen_pos),
			                  s);

			painter->drawLine(left,
			                  round(screen_pos),
			                  left - TICK_LENGTH,
			                  round(screen_pos));

		}

		screen_pos += screen_subtick_height;
		scene_pos += scene_subtick_height;
		++y_subtick_index;
	}

	painter->restore();
}

void PVParallelView::PVZoomableDrawingAreaWithAxes::draw_deco_v4(QPainter *painter,
                                                                 const QRectF &rect)
{
	int ticks_per_level = get_ticks_per_level();
	qreal log_ticks_per_level = log(ticks_per_level);

	int left = get_scene_left_margin();
	int right = left + get_x_axis_length();
	int top = get_scene_top_margin();
	int margin_bottom = get_y_axis_length();
	int bottom = top + margin_bottom;

	painter->save();
	painter->resetTransform();
	painter->setPen(_decoration_color);

	painter->drawLine(left, top, left, bottom);
	painter->drawLine(left, bottom, right, bottom);

	QFontMetrics fm = painter->fontMetrics();
	int fm_ascent = fm.ascent();

	std::cout << std::fixed << "#####################################################" << std::endl;
	qreal scene_left = map_to_scene(QPoint(left, 0)).x();
	qreal scene_right = map_to_scene(QPoint(right, 0)).x();

	qreal x_range = scene_right - scene_left;
	qreal nx_max = ceil(log(x_range) / log_ticks_per_level) - 1;
	qreal scene_subtick_width = floor(pow(ticks_per_level, nx_max)) / nx_max;
	print_s(scene_subtick_width);

	qreal x_scale = x_zoom_to_scale(get_x_axis_zoom().get_clamped_value());
	qreal screen_subtick_width = scene_subtick_width * x_scale;
	print_s(screen_subtick_width);

	int64_t x_subtick_index = ceil(scene_left / scene_subtick_width);
	qreal scene_pos = x_subtick_index * scene_subtick_width;
	qreal screen_pos = map_from_scene(QPointF(scene_pos, 0)).x();

	bool need_x_subticks = (screen_subtick_width > SUBTICK_MINSIZE);

	// let's draw
	while ((int)screen_pos <= right) {
		if ((x_subtick_index % ticks_per_level) != 0) {
			if (need_x_subticks) {
				painter->drawLine(round(screen_pos),
				                  bottom,
				                  round(screen_pos),
				                  bottom + SMALL_TICK_LENGTH);
			}
		} else {
			// QString s = QString::number(scene_pos, 'f', 2);
			QString s = get_x_value_at(scene_pos);
			int s_len = fm.boundingRect(s).width();

			painter->drawText(screen_pos - (s_len / 2),
			                  bottom + SCALE_VALUE_OFFSET + fm_ascent,
			                  s);
			painter->drawLine(round(screen_pos),
			                  bottom,
			                  round(screen_pos),
			                  bottom + TICK_LENGTH);
		}

		screen_pos += screen_subtick_width;
		scene_pos += scene_subtick_width;
		++x_subtick_index;
	}










	painter->restore();
}
