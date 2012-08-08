/**
 * \file PVZoomedParallelScene2.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvparallelview/PVZoomedParallelScene2.h>

#include <pvkernel/core/PVAlgorithms.h>


#include <QScrollBar>

/*****************************************************************************/

#define print_rect(R) _print_rect(#R, R)

template <typename T>
void _print_rect(const char* text, T r)
{
	std::cout << text << ": "
	          << r.x() << " " << r.y() << " "
	          << r.width() << " " << r.height() << std::endl;
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene2::PVZoomedParallelScene2
 *****************************************************************************/

PVParallelView::PVZoomedParallelScene2::PVZoomedParallelScene2(QWidget *parent,
                                                               zones_drawing_t &zones_drawing,
                                                               PVCol axis) :
	QGraphicsScene(parent),
	_zones_drawing(zones_drawing), _axis(axis)
{
	setBackgroundBrush(Qt::black);

	view()->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	view()->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
	view()->setResizeAnchor(QGraphicsView::AnchorViewCenter);
	view()->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
	view()->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);

	view()->setMaximumWidth(1024);
	view()->setMaximumHeight(1024);

	_wheel_value = 0;

	setSceneRect(-512, 0, 1024, 1024);

	update_zoom();

	if (axis > 0) {
		_left_image = zones_drawing.create_image(image_width);
	}

	if (axis < zones_drawing.get_zones_manager().get_number_zones()) {
		_right_image = zones_drawing.create_image(image_width);
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene2::mousePressEvent
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene2::mousePressEvent(QGraphicsSceneMouseEvent */*event*/)
{
	// kill default behaviour of QGraphicsScene's
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene2::mouseReleaseEvent
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene2::mouseReleaseEvent(QGraphicsSceneMouseEvent */*event*/)
{
	// kill default behaviour of QGraphicsScene's
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene2::mouseMoveEvent
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene2::mouseMoveEvent(QGraphicsSceneMouseEvent */*event*/)
{
	// kill default behaviour of QGraphicsScene's
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene2::wheelEvent
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene2::wheelEvent(QGraphicsSceneWheelEvent* event)
{
	if (event->modifiers() == Qt::ControlModifier) {
		// zoom
		if (event->delta() > 0) {
			if (_wheel_value < max_wheel_value) {
				++_wheel_value;
				update_zoom();
			}
		} else {
			if (_wheel_value > 0) {
				--_wheel_value;
				update_zoom();
			}
		}
	} else if (event->modifiers() == Qt::ShiftModifier) {
		// precise panning
		QScrollBar *sb = view()->verticalScrollBar();
		if (event->delta() > 0) {
			int v = sb->value();
			if (v > sb->minimum()) {
				sb->setValue(v - 1);
			}
		} else {
			int v = sb->value();
			if (v < sb->maximum()) {
				sb->setValue(v + 1);
			}
		}
	} else if (event->modifiers() == Qt::NoModifier) {
		// default panning
		QScrollBar *sb = view()->verticalScrollBar();
		if (event->delta() > 0) {
			sb->triggerAction(QAbstractSlider::SliderSingleStepSub);
		} else {
			sb->triggerAction(QAbstractSlider::SliderSingleStepAdd);
		}
	}

	event->accept();
}


/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene2::drawBackgroun
 *****************************************************************************/

/* TODO: add a mechanism to store BCI codes for a given area in scene to avoid
 * extracting them from the quadree each time drawBackground is called.
 */
void PVParallelView::PVZoomedParallelScene2::drawBackground(QPainter *painter,
                                                            const QRectF &/*rect*/)
{
	QRect screen_rect = view()->viewport()->rect();
	int screen_center = screen_rect.width() / 2;

	std::cout << "=============================" << std::endl;

	_back_image = QImage(screen_rect.size(), QImage::Format_ARGB32);
	_back_image.fill(Qt::black);

	QRectF screen_rect_s = view()->mapToScene(screen_rect).boundingRect();
	print_rect(screen_rect_s);

	QRectF view_rect = sceneRect().intersected(screen_rect_s);
	print_rect(view_rect);

	// true formula: UINT32_MAX * (x / 1024.0)
	uint32_t y_min = view_rect.top() * (UINT32_MAX >> 10);
	uint32_t y_max = view_rect.bottom() * (UINT32_MAX >> 10);

	std::cout << "_zoom_level: " << _zoom_level << std::endl;
	std::cout << "render from " << y_min << " to " << y_max
	          << " (" << y_max - y_min << ")" << std::endl;

	/* TODO: write the zones rendering
	 */

	// we need a painter to draw in _back_image
	QPainter image_painter(&_back_image);

	int step = get_zoom_step();

	int gap_y = (screen_rect_s.top() < 0)?round(-screen_rect_s.top()):0;
	std::cout << "gap_y: " << gap_y << std::endl;
	double alpha = 0.5 * pow(root_step, step);
	double beta = 1 / get_scale_factor();

	if (_left_image.get() != nullptr) {
		_zones_drawing.draw_zoomed_zone(*_left_image, y_min, _zoom_level, _axis - 1,
		                                &PVParallelView::PVZoomedZoneTree::browse_tree_bci_by_y2,
		                                alpha, beta);
		int gap_x = - PARALLELVIEW_AXIS_WIDTH / 2;

		image_painter.drawImage(QPoint(screen_center - gap_x - image_width, gap_y),
		                        _left_image->qimage());
	}

	if (_right_image.get() != nullptr) {
		_zones_drawing.draw_zoomed_zone(*_right_image, y_min, y_max, _zoom_level, _axis,
		                                &PVParallelView::PVZoomedZoneTree::browse_tree_bci_by_y1_range,
		                                alpha, beta);

		int value = 1 + screen_center + PARALLELVIEW_AXIS_WIDTH / 2;

		image_painter.drawImage(QPoint(value, gap_y),
		                        _right_image->qimage());
	}

	// we had to save the painter's state to restore it later
	// the scene transformation matrix is unneeded
	QTransform t = painter->transform();
	painter->resetTransform();

	// the pen has to be saved too
	QPen old_pen = painter->pen();

	// TODO: do the image stuff
	painter->drawImage(QPoint(0,0), _back_image);

	// draw axis
	QPen new_pen = QPen(Qt::white);
	new_pen.setColor(QColor(0xFFFFFFFF));
	new_pen.setWidth(PARALLELVIEW_AXIS_WIDTH);
	painter->setPen(new_pen);
	painter->drawLine(screen_center, 0, screen_center, screen_rect.height());

	// get back the painter's original state
	painter->setTransform(t);
	painter->setPen(old_pen);
}

