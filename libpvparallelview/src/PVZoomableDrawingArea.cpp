
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

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingArea::PVZoomableDrawingArea
 *****************************************************************************/

PVParallelView::PVZoomableDrawingArea::PVZoomableDrawingArea(QWidget *parent) :
	PVWidgets::PVGraphicsView(parent),
	_zoom_policy(PVParallelView::PVZoomableDrawingArea::AlongBoth),
	_zoom_min(0),
	_zoom_max(100),
	_zoom_value(0),
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
 * PVParallelView::PVZoomableDrawingArea::set_zoom_value
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingArea::set_zoom_value(const qint64 value)
{
	qint64 new_zoom_value = PVCore::clamp(value, _zoom_min, _zoom_max);

	if (new_zoom_value != _zoom_value) {
		_zoom_value = new_zoom_value;
		update_zoom();
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingArea::set_zoom_range
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingArea::set_zoom_range(const qint64 z_min,
                                                           const qint64 z_max)
{
	_zoom_min = z_min;
	_zoom_max = z_max;
	if(_zoom_value < z_min) {
		_zoom_value = z_min;
		update_zoom();
	} else if (_zoom_value > z_max) {
		_zoom_value = z_max;
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
 * PVParallelView::PVZoomableDrawingArea::scale_to_zoom
 *****************************************************************************/

int PVParallelView::PVZoomableDrawingArea::scale_to_zoom(const qreal scale_value) const
{
	return log(scale_value) / log(2.0);
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingArea::scale_to_zoom
 *****************************************************************************/

QTransform PVParallelView::PVZoomableDrawingArea::scale_to_transform(const qreal scale_value)
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
			transfo.scale(scale_value, v.height() / s.height());
		}
		break;
	case AlongY:
		{
			QRectF v = get_real_viewport_rect();
			QRectF s = get_scene_rect();
			transfo.scale(v.width() / s.width(), scale_value);
		}
		break;
	case AlongBoth:
		transfo.scale(scale_value, scale_value);
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
	} else if (event->modifiers() == SLOW_PAN_MODIFIER) {
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

	event->setAccepted(true);
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingArea::update_zoom
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingArea::update_zoom()
{
	QTransform transfo = scale_to_transform(zoom_to_scale(_zoom_value));

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
		break;
	}
}
