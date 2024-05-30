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

#include <pvparallelview/PVZoomableDrawingAreaInteractorMajorY.h>

#include <QScrollBar>

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaInteractorMajorY::PVZoomableDrawingAreaInteractorMajorY
 *****************************************************************************/

PVParallelView::PVZoomableDrawingAreaInteractorMajorY::PVZoomableDrawingAreaInteractorMajorY(
    PVWidgets::PVGraphicsView* parent)
    : PVParallelView::PVZoomableDrawingAreaInteractor(parent)
{
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaInteractorMajorY::mousePressEvent
 *****************************************************************************/
bool PVParallelView::PVZoomableDrawingAreaInteractorMajorY::mousePressEvent(
    PVParallelView::PVZoomableDrawingArea*, QMouseEvent* event)
{
	if (event->button() == Qt::RightButton) {
		_pan_reference = event->pos();
		event->setAccepted(true);
		return true;
	}

	return false;
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaInteractorMajorY::mouseReleaseEvent
 *****************************************************************************/

bool PVParallelView::PVZoomableDrawingAreaInteractorMajorY::mouseReleaseEvent(
    PVParallelView::PVZoomableDrawingArea*, QMouseEvent*)
{
	return false;
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaInteractorMajorY::mouseMoveEvent
 *****************************************************************************/

bool PVParallelView::PVZoomableDrawingAreaInteractorMajorY::mouseMoveEvent(
    PVParallelView::PVZoomableDrawingArea* zda, QMouseEvent* event)
{
	if (event->buttons() == Qt::RightButton) {

		QPoint delta = _pan_reference - event->pos();
		_pan_reference = event->pos();

		QScrollBar* sb;

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

bool PVParallelView::PVZoomableDrawingAreaInteractorMajorY::wheelEvent(
    PVParallelView::PVZoomableDrawingArea* zda, QWheelEvent* event)
{
	int mask = 0;

	if (event->modifiers() == Qt::NoModifier) {
		mask = PVParallelView::PVZoomableDrawingAreaConstraints::Y;
	} else if (event->modifiers() == Qt::ControlModifier) {
		mask = PVParallelView::PVZoomableDrawingAreaConstraints::X |
		       PVParallelView::PVZoomableDrawingAreaConstraints::Y;
	} else if (event->modifiers() == Qt::ShiftModifier) {
		mask = PVParallelView::PVZoomableDrawingAreaConstraints::X;
	}

	if (mask != 0) {
		int inc = (event->angleDelta().y() > 0) ? 1 : -1;

		event->setAccepted(true);

		if (increment_zoom_value(zda, mask, inc)) {
			zda->reconfigure_view();
			zda->get_viewport()->update();
			zoom_has_changed(zda, mask);
		}

		return true;
	}

	return false;
}
