
#include <pvparallelview/PVZoomableDrawingAreaInteractorMajorY.h>

#include <QScrollBar64>

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaInteractorMajorY::PVZoomableDrawingAreaInteractorMajorY
 *****************************************************************************/

PVParallelView::PVZoomableDrawingAreaInteractorMajorY::PVZoomableDrawingAreaInteractorMajorY(PVWidgets::PVGraphicsView* parent) :
	PVParallelView::PVZoomableDrawingAreaInteractor(parent)
{}

bool PVParallelView::PVZoomableDrawingAreaInteractorMajorY::mousePressEvent(PVParallelView::PVZoomableDrawingArea* zda, QMouseEvent* event)
{
	if (event->button() == Qt::RightButton) {
		_pan_reference = event->pos();
		event->setAccepted(true);
		return true;
	}

	return false;
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaInteractorMajorY::mouseMoveEvent
 *****************************************************************************/

bool PVParallelView::PVZoomableDrawingAreaInteractorMajorY::mouseMoveEvent(PVParallelView::PVZoomableDrawingArea* zda, QMouseEvent* event)
{
	if (event->buttons() == Qt::RightButton) {

		QPoint delta = _pan_reference - event->pos();
		_pan_reference = event->pos();

		QScrollBar64 *sb;

		sb = zda->get_horizontal_scrollbar();
		sb->setValue(sb->value() + delta.x());

		sb = zda->get_vertical_scrollbar();
		sb->setValue(sb->value() + delta.y());
		pan_has_changed(zda);
		event->setAccepted(true);
		return true;
	}

	return false;
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaInteractorMajorY::wheelEvent
 *****************************************************************************/

bool PVParallelView::PVZoomableDrawingAreaInteractorMajorY::wheelEvent(PVParallelView::PVZoomableDrawingArea* zda, QWheelEvent* event)
{
	int mask = 0;

	if (event->modifiers() == Qt::NoModifier) {
		mask = PVParallelView::PVZoomableDrawingAreaConstraints::Y;
	} else if (event->modifiers() == Qt::ControlModifier) {
		mask = PVParallelView::PVZoomableDrawingAreaConstraints::X | PVParallelView::PVZoomableDrawingAreaConstraints::Y;
	} else if (event->modifiers() == Qt::ShiftModifier) {
		mask = PVParallelView::PVZoomableDrawingAreaConstraints::X;
	}

	if (mask != 0) {
		int inc = (event->delta() > 0)?1:-1;

		event->setAccepted(true);

		if (increment_zoom_value(zda, mask, inc)) {
			zda->reconfigure_view();
			zda->update();
			zoom_has_changed(zda);
		}

		return true;
	}

	return false;
}
