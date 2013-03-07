
#include <pvkernel/core/PVAlgorithms.h>

#include <pvparallelview/PVZoomableDrawingAreaWithAxes.h>

#include <QString>
#include <QPainter>

#include <iostream>

#define SMALL_TICK_LENGTH 3
#define TICK_LENGTH (SMALL_TICK_LENGTH * 2)
#define SCALE_VALUE_OFFSET 8

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
	_ticks_count(10)
{
	set_scene_margins(50, 50, 40, 40);
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

void PVParallelView::PVZoomableDrawingAreaWithAxes::set_ticks_count(int count)
{
	if (_ticks_count != count) {
		_ticks_count = count;
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
	_y_axis_length = PVCore::min(scene_in_screen.height(), bottom - margin_top);

	int l = PVCore::max(painter->fontMetrics().boundingRect(get_y_value_at(-view_in_scene.y())).width(),
	                    painter->fontMetrics().boundingRect(get_y_value_at(-(view_in_scene.y() + view_in_scene.height()))).width());
	int r = painter->fontMetrics().boundingRect(get_y_value_at(-(view_in_scene.x() + view_in_scene.width()))).width();

	l += 2 * SCALE_VALUE_OFFSET;
	r = (r / 2) + SCALE_VALUE_OFFSET;

	if ((l > margin_left) || ( r > margin_right) || (margin_top != (bottom - _y_axis_length))) {
		set_scene_margins(l, r, bottom - _y_axis_length, margin_bottom);
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaWithAxes::draw_decorations
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingAreaWithAxes::draw_decorations(QPainter *painter,
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

#if 1
	// ticks
	qreal x_step = _x_axis_length / (qreal)_ticks_count;
	for(int i = 0; i <= _ticks_count; ++i) {
		qreal v = i * x_step;
		QString s = get_x_value_at(map_to_scene(left + i * x_step, 0).x());
		int s_len = fm.boundingRect(s).width();
		painter->drawLine(left + v, bottom, left + v, bottom + TICK_LENGTH);
		painter->drawText(left + v - (s_len / 2),
		                  bottom + SCALE_VALUE_OFFSET + fm_ascent,
		                  s);
	}
#else
	qreal ticks_gap = scene_in_screen.width() / (qreal)_ticks_count;
#endif

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
	qreal y_step = _y_axis_length / (qreal)_ticks_count;
	for(int i = 0; i <= _ticks_count; ++i) {
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
