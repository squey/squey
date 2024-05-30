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

#include <pvparallelview/PVZoomableDrawingAreaInteractorHomothetic.h>

#include <QScrollBar>

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaInteractorHomothetic::PVZoomableDrawingAreaInteractorHomothetic
 *****************************************************************************/

PVParallelView::PVZoomableDrawingAreaInteractorHomothetic::
    PVZoomableDrawingAreaInteractorHomothetic(PVWidgets::PVGraphicsView* parent)
    : PVParallelView::PVZoomableDrawingAreaInteractor(parent)
{
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaInteractorHomothetic::mousePressEvent
 *****************************************************************************/

bool PVParallelView::PVZoomableDrawingAreaInteractorHomothetic::mousePressEvent(
    PVParallelView::PVZoomableDrawingArea*, QMouseEvent* event)
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

bool PVParallelView::PVZoomableDrawingAreaInteractorHomothetic::mouseMoveEvent(
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
 * PVParallelView::PVZoomableDrawingAreaInteractorHomothetic::wheelEvent
 *****************************************************************************/

bool PVParallelView::PVZoomableDrawingAreaInteractorHomothetic::wheelEvent(
    PVParallelView::PVZoomableDrawingArea* zda, QWheelEvent* event)
{
	if (event->modifiers() == Qt::NoModifier) {
		int inc = (event->angleDelta().y() > 0) ? 1 : -1;
		bool ret = increment_zoom_value(
		    zda, PVZoomableDrawingAreaConstraints::X | PVZoomableDrawingAreaConstraints::Y, inc);
		event->setAccepted(true);

		if (ret) {
			zda->reconfigure_view();
			zoom_has_changed(zda, PVZoomableDrawingAreaConstraints::X |
			                          PVZoomableDrawingAreaConstraints::Y);
			zda->get_viewport()->update();
		}
	}

	return true;
}
