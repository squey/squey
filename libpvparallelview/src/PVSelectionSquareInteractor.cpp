
#include <pvparallelview/PVSelectionSquare.h>
#include <pvparallelview/PVSelectionSquareInteractor.h>
#include <pvparallelview/PVZoomableDrawingArea.h>

#include <QKeyEvent>
#include <QMouseEvent>

/*****************************************************************************
 * PVParallelView::PVSelectionSquareInteractor
 *****************************************************************************/

PVParallelView::PVSelectionSquareInteractor::PVSelectionSquareInteractor(PVWidgets::PVGraphicsView* parent,
                                                                         PVSelectionSquare* selection_square) :
	PVWidgets::PVGraphicsViewInteractor<PVWidgets::PVGraphicsView>(parent),
	_selection_square(selection_square)
{
	assert(selection_square->scene() == parent->get_scene());
}

/*****************************************************************************
 * PVParallelView::PVSelectionSquareInteractor::keyPressEvent
 *****************************************************************************/

bool PVParallelView::PVSelectionSquareInteractor::keyPressEvent(PVWidgets::PVGraphicsView* view,
                                                                QKeyEvent* event)
{
	bool x_axis_inverted = false;
	bool y_axis_inverted = false;

	if (PVZoomableDrawingArea* zda = dynamic_cast<PVZoomableDrawingArea*>(view)) {
		x_axis_inverted = zda->x_axis_inverted();
		y_axis_inverted = zda->y_axis_inverted();
	}

	if (event->key() == Qt::Key_Left) {
		if (event->modifiers() & Qt::ShiftModifier) {
			_selection_square->grow_horizontally();
		}
		else if (event->modifiers() & Qt::ControlModifier) {
			_selection_square->move_horizontally_by_width(!x_axis_inverted);
		}
		else {
			_selection_square->move_horizontally_by_step(!x_axis_inverted);
		}
		event->accept();
	}
	else if (event->key() == Qt::Key_Right) {
		if (event->modifiers() & Qt::ShiftModifier) {
			_selection_square->shrink_horizontally();
		}
		else if (event->modifiers() & Qt::ControlModifier) {
			_selection_square->move_horizontally_by_width(x_axis_inverted);
		}
		else {
			_selection_square->move_horizontally_by_step(x_axis_inverted);
		}
		event->accept();
	}
	else if (event->key() == Qt::Key_Up) {
		if (event->modifiers() & Qt::ShiftModifier) {
			_selection_square->grow_vertically();
		}
		else if (event->modifiers() & Qt::ControlModifier) {
			_selection_square->move_vertically_by_height(!y_axis_inverted);
		}
		else {
			_selection_square->move_vertically_by_step(!y_axis_inverted);
		}
		event->accept();
	}
	else if (event->key() == Qt::Key_Down) {
		if (event->modifiers() & Qt::ShiftModifier) {
			_selection_square->shrink_vertically();
		}
		else if (event->modifiers() & Qt::ControlModifier) {
			_selection_square->move_vertically_by_height(y_axis_inverted);
		}
		else {
			_selection_square->move_vertically_by_step(y_axis_inverted);
		}
		event->accept();
	}

	return false;
}

/*****************************************************************************
 * PVParallelView::PVSelectionSquareInteractor::mousePressEvent
 *****************************************************************************/

bool PVParallelView::PVSelectionSquareInteractor::mousePressEvent(PVWidgets::PVGraphicsView* view,
                                                                  QMouseEvent* event)
{
	if (event->button() == Qt::LeftButton) {
		QPointF p = view->map_to_scene(event->pos());
		_selection_square->begin(p.x(), p.y());
		event->accept();
	}
	return false;
}

/*****************************************************************************
 * PVParallelView::PVSelectionSquareInteractor::mouseReleaseEvent
 *****************************************************************************/

bool PVParallelView::PVSelectionSquareInteractor::mouseReleaseEvent(PVWidgets::PVGraphicsView* view,
                                                                    QMouseEvent* event)
{
	if (event->button() == Qt::LeftButton) {
		QPointF p = view->map_to_scene(event->pos());
		_selection_square->end(p.x(), p.y(), true, true);
		event->accept();
	}
	return false;
}

/*****************************************************************************
 * PVParallelView::PVSelectionSquareInteractor::mouseMoveEvent
 *****************************************************************************/

bool PVParallelView::PVSelectionSquareInteractor::mouseMoveEvent(PVWidgets::PVGraphicsView* view,
                                                                 QMouseEvent* event)
{
	if (event->buttons() == Qt::LeftButton)	{
		QPointF p = view->map_to_scene(event->pos());
		_selection_square->end(p.x(), p.y());
		event->accept();
	}

	return false;
}
