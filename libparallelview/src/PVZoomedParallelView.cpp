
#include <iostream>

#include <pvparallelview/PVZoomedParallelView.h>
#include <pvparallelview/PVZonesDrawing.h>
#include <pvparallelview/PVZoomedZoneView.h>

PVParallelView::PVZoomedParallelView::PVZoomedParallelView(QObject* parent,
                                                           PVParallelView::PVZonesDrawing &zones_drawing,
                                                           PVCol axis) :
	QGraphicsScene(parent),
	_zones_drawing(zones_drawing),
	_left_zone(0),
	_right_zone(0),
	_axis(axis),
	_y_position(0)
{
	setBackgroundBrush(Qt::black);

	if (axis > 0) {
		_left_zone = new PVParallelView::PVZoomedZoneView(zones_drawing, axis - 1, false);
	}
	if (axis < zones_drawing.get_zones_manager().get_number_cols()) {
		_right_zone = new PVParallelView::PVZoomedZoneView(zones_drawing, axis, true);
	}
}

PVParallelView::PVZoomedParallelView::~PVZoomedParallelView()
{
	if (_left_zone) {
		delete _left_zone;
	}
	if (_right_zone) {
		delete _right_zone;
	}
}

void PVParallelView::PVZoomedParallelView::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
	if (event->buttons() == Qt::RightButton) {
		int delta = (int)(_translation_start_y - event->scenePos().y());
		std::cout << "PVZoomedParallelView:mouseMoveEvent() delta: "
		          << delta << std::endl;
	}
}

void PVParallelView::PVZoomedParallelView::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
	if (event->button() == Qt::RightButton) {
		_translation_start_y = event->scenePos().y();
		std::cout << "PVZoomedParallelView:mousePressEvent() start_y: "
		          << _translation_start_y << std::endl;
	}
}

void PVParallelView::PVZoomedParallelView::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
	std::cout << "PVZoomedParallelView:mouseReleaseEvent()" << std::endl;
	if (event->button() == Qt::RightButton) {
	}
}

void PVParallelView::PVZoomedParallelView::wheelEvent(QGraphicsSceneWheelEvent* event)
{
	std::cout << "PVZoomedParallelView:wheelEvent()" << std::endl;

	// int zoom = event->delta() / 2;
	if (event->modifiers() == Qt::ControlModifier) {
	} else{
	}
}
