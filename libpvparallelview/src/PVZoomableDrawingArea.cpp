
#include <pvkernel/core/PVAlgorithms.h>

#include <pvparallelview/PVZoomableDrawingArea.h>

#include <QGraphicsScene>
#include <QScrollBar64>
#include <QMouseEvent>
#include <QWheelEvent>

#include <iostream>

#define ZOOM_MODIFIER     Qt::NoModifier
#define PAN_MODIFIER      Qt::ControlModifier
#define SLOW_PAN_MODIFIER Qt::ShiftModifier

#define SYNCHRONIZED_ZOOM_MODIFIER Qt::ControlModifier
#define SECONDARY_ZOOM_MODIFIER Qt::ShiftModifier

#define UnBindMask 4

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingArea::PVZoomableDrawingArea
 *****************************************************************************/

PVParallelView::PVZoomableDrawingArea::PVZoomableDrawingArea(QWidget *parent) :
	PVWidgets::PVGraphicsView(parent),
	_zoom_policy(PVParallelView::PVZoomableDrawingArea::AlongBoth),
	_x_zoom_min(0),
	_x_zoom_max(100),
	_x_zoom_value(0),
	_y_zoom_min(0),
	_y_zoom_max(100),
	_y_zoom_value(0),
	_pan_policy(PVParallelView::PVZoomableDrawingArea::AlongBoth)
{
	set_transformation_anchor(AnchorUnderMouse);
	set_resize_anchor(AnchorViewCenter);

	QGraphicsScene *scene = new QGraphicsScene();
	set_scene(scene);
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingArea::~PVZoomableDrawingArea
 *****************************************************************************/

PVParallelView::PVZoomableDrawingArea::~PVZoomableDrawingArea()
{
	get_scene()->deleteLater();
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingArea::adjust_zoom_values
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingArea::adjust_zoom_values()
{
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingArea::set_zoom_value
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingArea::set_zoom_value(const qint64 value,
                                                           const bool propagate_update)
{
	bool need_update = false;
	qint64 new_zoom_value = PVCore::clamp(value, _x_zoom_min, _x_zoom_max);

	if (new_zoom_value != _x_zoom_value) {
		_x_zoom_value = new_zoom_value;
		need_update |= true;
	}

	new_zoom_value = PVCore::clamp(value, _x_zoom_min, _x_zoom_max);

	if (new_zoom_value != _y_zoom_value) {
		_y_zoom_value = new_zoom_value;
		need_update |= true;
	}

	if (need_update && propagate_update) {
		update_zoom();
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingArea::set_x_zoom_value
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingArea::set_x_zoom_value(const qint64 value,
                                                             const bool propagate_update)
{
	qint64 new_zoom_value = PVCore::clamp(value, _x_zoom_min, _x_zoom_max);

	if (new_zoom_value != _x_zoom_value) {
		_x_zoom_value = new_zoom_value;
		if (propagate_update) {
			update_zoom();
		}
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingArea::set_y_zoom_value
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingArea::set_y_zoom_value(const qint64 value,
                                                             const bool propagate_update)
{
	qint64 new_zoom_value = PVCore::clamp(value, _y_zoom_min, _y_zoom_max);

	if (new_zoom_value != _y_zoom_value) {
		_y_zoom_value = new_zoom_value;
		if (propagate_update) {
			update_zoom();
		}
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingArea::set_zoom_range
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingArea::set_zoom_range(const qint64 z_min,
                                                           const qint64 z_max,
                                                           const bool propagate_update)
{
	bool need_update = false;

	_x_zoom_min = _y_zoom_min = z_min;
	_x_zoom_max = _y_zoom_max = z_max;

	if(_x_zoom_value < z_min) {
		_x_zoom_value = z_min;
		need_update |= true;
	} else if (_x_zoom_value > z_max) {
		_x_zoom_value = z_max;
		need_update |= true;
	}

	if(_y_zoom_value < z_min) {
		_y_zoom_value = z_min;
		need_update |= true;
	} else if (_y_zoom_value > z_max) {
		_y_zoom_value = z_max;
		need_update |= true;
	}

	if (need_update && propagate_update) {
		update_zoom();
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingArea::set_x_zoom_range
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingArea::set_x_zoom_range(const qint64 z_min,
                                                             const qint64 z_max,
                                                             const bool propagate_update)
{
	bool need_update = false;

	_x_zoom_min = z_min;
	_x_zoom_max = z_max;

	if(_x_zoom_value < z_min) {
		_x_zoom_value = z_min;
		need_update |= true;
	} else if (_x_zoom_value > z_max) {
		_x_zoom_value = z_max;
		need_update |= true;
	}

	if (need_update && propagate_update) {
		update_zoom();
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingArea::set_y_zoom_range
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingArea::set_y_zoom_range(const qint64 z_min,
                                                             const qint64 z_max,
                                                             const bool propagate_update)
{
	bool need_update = false;

	_y_zoom_min = z_min;
	_y_zoom_max = z_max;
	if(_y_zoom_value < z_min) {
		_y_zoom_value = z_min;
		need_update |= true;
	} else if (_y_zoom_value > z_max) {
		_y_zoom_value = z_max;
		need_update |= true;
	}

	if (need_update && propagate_update) {
		update_zoom();
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingArea::zoom_to_scale
 *****************************************************************************/

qreal PVParallelView::PVZoomableDrawingArea::zoom_to_scale(const int zoom_value) const
{
	return pow(2.0, zoom_value);
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingArea::x_zoom_to_scale
 *****************************************************************************/

qreal PVParallelView::PVZoomableDrawingArea::x_zoom_to_scale(const int zoom_value) const
{
	return PVZoomableDrawingArea::zoom_to_scale(zoom_value);
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingArea::y_zoom_to_scale
 *****************************************************************************/

qreal PVParallelView::PVZoomableDrawingArea::y_zoom_to_scale(const int zoom_value) const
{
	return PVZoomableDrawingArea::zoom_to_scale(zoom_value);
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingArea::scale_to_zoom
 *****************************************************************************/

int PVParallelView::PVZoomableDrawingArea::scale_to_zoom(const qreal scale_value) const
{
	return log(scale_value) / log(2.0);
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingArea::x_scale_to_zoom
 *****************************************************************************/

int PVParallelView::PVZoomableDrawingArea::x_scale_to_zoom(const qreal scale_value) const
{
	return PVZoomableDrawingArea::scale_to_zoom(scale_value);
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingArea::y_scale_to_zoom
 *****************************************************************************/

int PVParallelView::PVZoomableDrawingArea::y_scale_to_zoom(const qreal scale_value) const
{
	return PVZoomableDrawingArea::scale_to_zoom(scale_value);
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingArea::scale_to_transform
 *****************************************************************************/

QTransform PVParallelView::PVZoomableDrawingArea::scale_to_transform(const qreal x_scale_value,
                                                                     const qreal y_scale_value) const
{
	QTransform transfo;

	switch(_zoom_policy) {
	case AlongNone:
		{
			QRectF v = get_real_viewport_rect();
			QRectF s = get_scene_rect();
			transfo.scale(v.width() / s.width(), v.height() / s.height());
		}
		break;
	case AlongX:
		{
			QRectF v = get_real_viewport_rect();
			QRectF s = get_scene_rect();
			transfo.scale(x_scale_value, v.height() / s.height());
		}
		break;
	case AlongY:
		{
			QRectF v = get_real_viewport_rect();
			QRectF s = get_scene_rect();
			transfo.scale(v.width() / s.width(), y_scale_value);
		}
		break;
	case AlongBoth:
	case BoundMajorX:
	case BoundMajorY:
		transfo.scale(x_scale_value, y_scale_value);
		break;
	}

	return transfo;
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingArea::mousePressEvent
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingArea::mousePressEvent(QMouseEvent *event)
{
	// do parent's call to grab hover events
	PVWidgets::PVGraphicsView::mousePressEvent(event);

	if (event->button() == Qt::RightButton) {
		_pan_reference = event->pos();
		event->accept();
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingArea::mouseReleaseEvent
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingArea::mouseReleaseEvent(QMouseEvent *event)
{
	// implement it in case of
	// do parent's call to grab hover events
	PVWidgets::PVGraphicsView::mouseReleaseEvent(event);
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingArea::mouseMoveEvent
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingArea::mouseMoveEvent(QMouseEvent *event)
{
	// do parent's call to grab hover events
	PVWidgets::PVGraphicsView::mouseMoveEvent(event);

	if (event->buttons() == Qt::RightButton) {
		bool has_moved = false;
		QPoint delta = _pan_reference - event->pos();
		_pan_reference = event->pos();

		QScrollBar64 *sb;

		if (_pan_policy & AlongX) {
			sb = get_horizontal_scrollbar();
			sb->setValue(sb->value() + delta.x());
			has_moved |= true;
		}

		if (_pan_policy & AlongY) {
			sb = get_vertical_scrollbar();
			sb->setValue(sb->value() + delta.y());
			has_moved |= true;
		}

		if (has_moved) {
			emit pan_has_changed();
		}

		event->accept();
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingArea::resizeEvent
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingArea::resizeEvent(QResizeEvent *event)
{
	PVWidgets::PVGraphicsView::resizeEvent(event);
	update_zoom();
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingArea::mousePressEvent
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingArea::wheelEvent(QWheelEvent *event)
{
	bool has_zoomed = false;

#if 0
	if (event->modifiers() == ZOOM_MODIFIER) {
		// zoom
		if (event->delta() > 0) {
			if (_zoom_value < _zoom_max) {
				++_zoom_value;
				update_zoom();
				emit zoom_has_changed();
			}
		} else {
			if (_zoom_value > _zoom_min) {
				--_zoom_value;
				update_zoom();
				emit zoom_has_changed();
			}
		}
	}
	else if (event->modifiers() == SLOW_PAN_MODIFIER) {
		// precise panning
		QScrollBar64 *sb = get_vertical_scrollbar();
		if (event->delta() > 0) {
			qint64 v = sb->value();
			if (v > sb->minimum()) {
				sb->setValue(v - 1);
				emit pan_has_changed();
			}
		} else {
			qint64 v = sb->value();
			if (v < sb->maximum()) {
				sb->setValue(v + 1);
				emit pan_has_changed();
			}

		}
	} else if (event->modifiers() == PAN_MODIFIER) {
		// panning
		QScrollBar64 *sb = get_vertical_scrollbar();
		if (event->delta() > 0) {
			sb->triggerAction(QAbstractSlider64::SliderSingleStepSub);
			emit pan_has_changed();
		} else {
			sb->triggerAction(QAbstractSlider64::SliderSingleStepAdd);
			emit pan_has_changed();
		}
	}
#endif

	if (has_zoomed) {
		update_zoom();
		emit zoom_has_changed();
	}

	event->setAccepted(true);
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingArea::update_zoom
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingArea::update_zoom()
{
	adjust_zoom_values();

	QTransform transfo = scale_to_transform(x_zoom_to_scale(_x_zoom_value),
	                                        y_zoom_to_scale(_y_zoom_value));

	set_transform(transfo);

	adjust_pan();

	update();
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingArea::adjust_pan
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingArea::adjust_pan()
{
	switch(_pan_policy) {
	case AlongNone:
		break;
	case AlongX:
		{
			QScrollBar64 *sb = get_vertical_scrollbar();
			int64_t mid = ((int64_t)sb->maximum() + sb->minimum()) / 2;
			sb->setValue(mid);
		}
		break;
	case AlongY:
		{
			QScrollBar64 *sb = get_horizontal_scrollbar();
			int64_t mid = ((int64_t)sb->maximum() + sb->minimum()) / 2;
			sb->setValue(mid);
		}
		break;
	case AlongBoth:
	case BoundMajorX:
	case BoundMajorY:
		break;
	}
}
