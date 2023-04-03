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

#include <pvparallelview/PVSelectionRectangle.h>
#include <pvparallelview/PVSelectionRectangleInteractor.h>
#include <pvparallelview/PVZoomableDrawingArea.h>

#include <QKeyEvent>
#include <QMouseEvent>

/*****************************************************************************
 * PVParallelView::PVSelectionRectangleInteractor
 *****************************************************************************/

PVParallelView::PVSelectionRectangleInteractor::PVSelectionRectangleInteractor(
    PVWidgets::PVGraphicsView* parent, PVSelectionRectangle* selection_rectangle)
    : PVWidgets::PVGraphicsViewInteractor<PVWidgets::PVGraphicsView>(parent)
    , _selection_rectangle(selection_rectangle)
{
	assert(selection_rectangle->scene() == parent->get_scene());
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangleInteractor::keyPressEvent
 *****************************************************************************/

bool PVParallelView::PVSelectionRectangleInteractor::keyPressEvent(PVWidgets::PVGraphicsView* view,
                                                                   QKeyEvent* event)
{
	bool x_axis_inverted = false;
	bool y_axis_inverted = false;

	if (auto* zda = dynamic_cast<PVZoomableDrawingArea*>(view)) {
		x_axis_inverted = zda->x_axis_inverted();
		y_axis_inverted = zda->y_axis_inverted();
	}

	switch (event->key()) {
	/**
	 * RH: as the PVFullParallelView can be not used with the PVGraphicsView's
	 * interactor mechanism, this code is voluntary replicated in the method
	 * PVFulParallelScene::keyPressEvent(...)
	 */
	case Qt::Key_Left:
		if (event->modifiers() == Qt::ShiftModifier) {
			_selection_rectangle->grow_horizontally();
			event->accept();
		} else if (event->modifiers() == Qt::ControlModifier) {
			_selection_rectangle->move_horizontally_by_width(!x_axis_inverted);
			event->accept();
		} else if (event->modifiers() == Qt::NoModifier) {
			_selection_rectangle->move_horizontally_by_step(!x_axis_inverted);
			event->accept();
		}
		break;
	case Qt::Key_Right:
		if (event->modifiers() == Qt::ShiftModifier) {
			_selection_rectangle->shrink_horizontally();
			event->accept();
		} else if (event->modifiers() == Qt::ControlModifier) {
			_selection_rectangle->move_horizontally_by_width(x_axis_inverted);
			event->accept();
		} else if (event->modifiers() == Qt::NoModifier) {
			_selection_rectangle->move_horizontally_by_step(x_axis_inverted);
			event->accept();
		}
		break;
	case Qt::Key_Up:
		if (event->modifiers() == Qt::ShiftModifier) {
			_selection_rectangle->grow_vertically();
			event->accept();
		} else if (event->modifiers() == Qt::ControlModifier) {
			_selection_rectangle->move_vertically_by_height(!y_axis_inverted);
			event->accept();
		} else if (event->modifiers() == Qt::NoModifier) {
			_selection_rectangle->move_vertically_by_step(!y_axis_inverted);
			event->accept();
		}
		break;
	case Qt::Key_Down:
		if (event->modifiers() == Qt::ShiftModifier) {
			_selection_rectangle->shrink_vertically();
			event->accept();
		} else if (event->modifiers() == Qt::ControlModifier) {
			_selection_rectangle->move_vertically_by_height(y_axis_inverted);
			event->accept();
		} else if (event->modifiers() == Qt::NoModifier) {
			_selection_rectangle->move_vertically_by_step(y_axis_inverted);
			event->accept();
		}
		break;
	}

	if (event->isAccepted()) {
		view->fake_mouse_move();
		view->get_viewport()->update();
	}

	return false;
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangleInteractor::mousePressEvent
 *****************************************************************************/

bool PVParallelView::PVSelectionRectangleInteractor::mousePressEvent(
    PVWidgets::PVGraphicsView* view, QMouseEvent* event)
{
	if (event->button() == Qt::LeftButton) {
		QPointF p = view->map_to_scene(event->pos());
		_selection_rectangle->begin(p);
		event->accept();
	}
	return false;
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangleInteractor::mouseReleaseEvent
 *****************************************************************************/

bool PVParallelView::PVSelectionRectangleInteractor::mouseReleaseEvent(
    PVWidgets::PVGraphicsView* view, QMouseEvent* event)
{
	if (event->button() == Qt::LeftButton) {
		QPointF p = view->map_to_scene(event->pos());
		_selection_rectangle->end(p);
		if (_selection_rectangle->get_rect().isNull()) {
			_selection_rectangle->clear();
		}
		view->fake_mouse_move();
		event->accept();
	}
	return false;
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangleInteractor::mouseMoveEvent
 *****************************************************************************/

bool PVParallelView::PVSelectionRectangleInteractor::mouseMoveEvent(PVWidgets::PVGraphicsView* view,
                                                                    QMouseEvent* event)
{
	if (event->buttons() == Qt::LeftButton) {
		QPointF p = view->map_to_scene(event->pos());
		_selection_rectangle->step(p);
		event->accept();
	}

	return false;
}
