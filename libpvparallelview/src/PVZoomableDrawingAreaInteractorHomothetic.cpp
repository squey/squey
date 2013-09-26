
#include <pvparallelview/PVZoomableDrawingAreaInteractorHomothetic.h>

#include <QScrollBar64>

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaInteractorHomothetic::PVZoomableDrawingAreaInteractorHomothetic
 *****************************************************************************/

PVParallelView::PVZoomableDrawingAreaInteractorHomothetic::PVZoomableDrawingAreaInteractorHomothetic(PVWidgets::PVGraphicsView* parent) :
	PVParallelView::PVZoomableDrawingAreaInteractor(parent)
{}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaInteractorHomothetic::mousePressEvent
 *****************************************************************************/

bool PVParallelView::PVZoomableDrawingAreaInteractorHomothetic::mousePressEvent(PVParallelView::PVZoomableDrawingArea*, QMouseEvent* event)
{
	if (event->button() == Qt::RightButton) {
		_pan_reference = event->pos();
		event->setAccepted(true);
	}

	return false;
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaInteractorHomothetic::mouseMoveEvent
 *****************************************************************************/

bool PVParallelView::PVZoomableDrawingAreaInteractorHomothetic::mouseMoveEvent(PVParallelView::PVZoomableDrawingArea* zda, QMouseEvent* event)
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
 * PVParallelView::PVZoomableDrawingAreaInteractorHomothetic::wheelEvent
 *****************************************************************************/

bool PVParallelView::PVZoomableDrawingAreaInteractorHomothetic::wheelEvent(PVParallelView::PVZoomableDrawingArea* zda, QWheelEvent* event)
{
	if (event->modifiers() == Qt::NoModifier) {
		int inc = (event->delta() > 0)?1:-1;
		bool ret = increment_zoom_value(zda,
		                                PVZoomableDrawingAreaConstraints::X
		                                |
		                                PVZoomableDrawingAreaConstraints::Y,
		                                inc);
		event->setAccepted(true);

		if (ret) {
			zda->reconfigure_view();
			zoom_has_changed(zda,
			                 PVZoomableDrawingAreaConstraints::X
			                 |
			                 PVZoomableDrawingAreaConstraints::Y);
			zda->get_viewport()->update();
		}
	}

	return true;
}
