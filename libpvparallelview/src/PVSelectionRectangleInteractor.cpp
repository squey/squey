
#include <pvparallelview/PVSelectionSquare.h>
#include <pvparallelview/PVSelectionRectangleInteractor.h>

#include <QKeyEvent>
#include <QMouseEvent>

/*****************************************************************************
 * PVParallelView::PVSelectionRectangleInteractor
 *****************************************************************************/

PVParallelView::PVSelectionRectangleInteractor::PVSelectionRectangleInteractor(PVWidgets::PVGraphicsView* parent,
                                                                               PVSelectionSquare* selection_rectangle) :
	PVWidgets::PVGraphicsViewInteractor<PVWidgets::PVGraphicsView>(parent),
	_selection_rectangle(selection_rectangle)
{
	assert(selection_rectangle->scene() == parent->get_scene());
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangleInteractor::keyPressEvent
 *****************************************************************************/

bool PVParallelView::PVSelectionRectangleInteractor::keyPressEvent(PVWidgets::PVGraphicsView* view,
                                                                   QKeyEvent* event)
{
	if (event->key() == Qt::Key_Left) {
		if (event->modifiers() & Qt::ShiftModifier) {
			_selection_rectangle->grow_horizontally();
		}
		else if (event->modifiers() & Qt::ControlModifier) {
			_selection_rectangle->move_left_by_width();
		}
		else {
			_selection_rectangle->move_left_by_step();
		}
		event->accept();
	}
	else if (event->key() == Qt::Key_Right) {
		if (event->modifiers() & Qt::ShiftModifier) {
			_selection_rectangle->shrink_horizontally();
		}
		else if (event->modifiers() & Qt::ControlModifier) {
			_selection_rectangle->move_right_by_width();
		}
		else {
			_selection_rectangle->move_right_by_step();
		}
		event->accept();
	}
	else if (event->key() == Qt::Key_Up) {
		if (event->modifiers() & Qt::ShiftModifier) {
			_selection_rectangle->grow_vertically();
		}
		else if (event->modifiers() & Qt::ControlModifier) {
			_selection_rectangle->move_up_by_height();
		}
		else {
			_selection_rectangle->move_up_by_step();
		}
		event->accept();
	}
	else if (event->key() == Qt::Key_Down) {
		if (event->modifiers() & Qt::ShiftModifier) {
			_selection_rectangle->shrink_vertically();
		}
		else if (event->modifiers() & Qt::ControlModifier) {
			_selection_rectangle->move_down_by_height();
		}
		else {
			_selection_rectangle->move_down_by_step();
		}
		event->accept();
	}

	return false;
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangleInteractor::mousePressEvent
 *****************************************************************************/

bool PVParallelView::PVSelectionRectangleInteractor::mousePressEvent(PVWidgets::PVGraphicsView* view,
                                                                     QMouseEvent* event)
{
	if (event->button() == Qt::LeftButton) {
		QPointF p = view->map_to_scene(event->pos());
		_selection_rectangle->begin(p.x(), p.y());
		event->accept();
	}
	return false;
}

/*****************************************************************************
 * PVParallelView::PVSelectionRectangleInteractor::mouseReleaseEvent
 *****************************************************************************/

bool PVParallelView::PVSelectionRectangleInteractor::mouseReleaseEvent(PVWidgets::PVGraphicsView* view,
	 QMouseEvent* event)
{
	if (event->button() == Qt::LeftButton) {
		QPointF p = view->map_to_scene(event->pos());
		_selection_rectangle->end(p.x(), p.y(), true, true);
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
	if (event->buttons() == Qt::LeftButton)	{
		QPointF p = view->map_to_scene(event->pos());
		_selection_rectangle->end(p.x(), p.y());
		event->accept();
	}

	return false;
}
