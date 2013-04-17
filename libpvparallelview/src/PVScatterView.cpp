/**
 * \file PVScatterView.cpp
 *
 * Copyright (C) Picviz Labs 2013
 */

#include <pvparallelview/PVScatterView.h>

#include <stdlib.h>     // for rand()

#include <QApplication>
#include <QGraphicsScene>
#include <QPainter>

#include <picviz/PVView.h>

#include <pvhive/PVCallHelper.h>
#include <pvhive/PVHive.h>
#include <pvhive/PVObserverCallback.h>

#include <pvparallelview/PVZoneTree.h>
#include <pvparallelview/PVBCode.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVSelectionSquareScatterView.h>
#include <pvparallelview/PVZoomableDrawingAreaInteractor.h>

namespace PVParallelView
{

class PVScatterViewInteractor : public PVWidgets::PVGraphicsViewInteractor<PVWidgets::PVGraphicsView>
{
	typedef PVWidgets::PVGraphicsViewInteractor<PVWidgets::PVGraphicsView> parent_type;

public:
	PVScatterViewInteractor(PVWidgets::PVGraphicsView* parent, Picviz::PVView& view, const PVZoneTree &zt) :
		parent_type(parent),
		_view(view),
		_scatter_view(static_cast<PVScatterView*>(parent)),
		_selection_square(new PVSelectionSquareScatterView(view, zt, _scatter_view->get_scene()))
	{}

	bool keyPressEvent(PVWidgets::PVGraphicsView* zda, QKeyEvent* event) override
	{
		if (event->key() == Qt::Key_Left) {
			if (event->modifiers() & Qt::ShiftModifier) {
				_selection_square->grow_horizontally();
			}
			else if (event->modifiers() & Qt::ControlModifier) {
				_selection_square->move_left_by_width();
			}
			else {
				_selection_square->move_left_by_step();
			}
			event->accept();
		}
		else if (event->key() == Qt::Key_Right) {
			if (event->modifiers() & Qt::ShiftModifier) {
				_selection_square->shrink_horizontally();
			}
			else if (event->modifiers() & Qt::ControlModifier) {
				_selection_square->move_right_by_width();
			}
			else {
				_selection_square->move_right_by_step();
			}
			event->accept();
		}
		else if (event->key() == Qt::Key_Up) {
			if (event->modifiers() & Qt::ShiftModifier) {
				_selection_square->grow_vertically();
			}
			else if (event->modifiers() & Qt::ControlModifier) {
				_selection_square->move_up_by_height();
			}
			else {
				_selection_square->move_up_by_step();
			}
			event->accept();
		}
		else if (event->key() == Qt::Key_Down) {
			if (event->modifiers() & Qt::ShiftModifier) {
				_selection_square->shrink_vertically();
			}
			else if (event->modifiers() & Qt::ControlModifier) {
				_selection_square->move_down_by_height();
			}
			else {
				_selection_square->move_down_by_step();
			}
			event->accept();
		}

		return false;
	}

	bool mousePressEvent(PVWidgets::PVGraphicsView* obj, QMouseEvent* event) override
	{
		if (event->button() == Qt::LeftButton) {
			_selection_square->begin(event->pos().x()-_scatter_view->get_real_viewport_rect().x(), event->pos().y()-_scatter_view->get_real_viewport_rect().y());
			event->accept();
		}
		return false;
	}

	bool mouseReleaseEvent(PVWidgets::PVGraphicsView* obj, QMouseEvent* event) override
	{
		if (event->button() == Qt::LeftButton) {
			_selection_square->end(event->pos().x()-_scatter_view->get_real_viewport_rect().x(), event->pos().y()-_scatter_view->get_real_viewport_rect().y(), true, true);
			event->accept();
		}
		return false;
	}

	bool mouseMoveEvent(PVWidgets::PVGraphicsView* obj, QMouseEvent* event) override
	{
		if (event->buttons() == Qt::LeftButton)
		{
			_selection_square->end(event->pos().x()-_scatter_view->get_real_viewport_rect().x(), event->pos().y()-_scatter_view->get_real_viewport_rect().y());
			event->accept();
		}
		return false;
	}

private:
	Picviz::PVView& _view;
	PVScatterView* _scatter_view;
	PVSelectionSquare* _selection_square;
	PVHive::PVActor<Picviz::PVView> _view_actor;
};

}

PVParallelView::PVScatterView::PVScatterView(
	const Picviz::PVView_sp &pvview_sp,
	const PVZoneTree &zt,
	QWidget* parent /*= nullptr*/
) :
	PVZoomableDrawingAreaWithAxes(parent),
	_view(*pvview_sp),
	_zt(zt),
	_view_deleted(false)
{
	setCursor(Qt::CrossCursor);
	//QRectF r(0, -(1L << 32), (1L << 32), (1L << 32));
	//get_scene()->setSceneRect(r);
	get_scene()->setSceneRect(0, 0, 1024, 1024);

	// interactor/constraints
	PVWidgets::PVGraphicsViewInteractorBase* inter = declare_interactor<PVScatterViewInteractor>(_view, _zt);
	register_front_all(inter);
	register_front_one(QEvent::Resize, inter);
	register_front_one(QEvent::KeyPress, inter);
	install_default_scene_interactor();
}

/*****************************************************************************
 * PVParallelView::PVScatterView::~PVScatterView
 *****************************************************************************/

PVParallelView::PVScatterView::~PVScatterView()
{
	if (!_view_deleted) {
		common::get_lib_view(_view)->remove_scatter_view(this);
	}
}

/*****************************************************************************
 * PVParallelView::PVScatterView::about_to_be_deleted
 *****************************************************************************/
void PVParallelView::PVScatterView::about_to_be_deleted()
{
	_view_deleted = true;
}

/*****************************************************************************
 * PVParallelView::PVScatterView::update_new_selection_async
 *****************************************************************************/
void PVParallelView::PVScatterView::update_new_selection_async()
{
	QMetaObject::invokeMethod(this, "update", Qt::QueuedConnection);
}

/*****************************************************************************
 * PVParallelView::PVScatterView::update_all_async
 *****************************************************************************/
void PVParallelView::PVScatterView::update_all_async()
{
	QMetaObject::invokeMethod(this, "update", Qt::QueuedConnection);
}

/*****************************************************************************
 * PVParallelView::PVScatterView::drawBackground
 *****************************************************************************/
void PVParallelView::PVScatterView::drawBackground(QPainter* painter, const QRectF& rect)
{
	painter->fillRect(rect, QColor::fromRgbF(0.1, 0.1, 0.1, 1.0));
	PVZoomableDrawingAreaWithAxes::drawBackground(painter, rect);
	draw_points(painter);
}

/*****************************************************************************
 * PVParallelView::PVScatterView::draw_points
 *****************************************************************************/
void PVParallelView::PVScatterView::draw_points(QPainter *painter)
{
	PVParallelView::PVBCode code_b;

	Picviz::PVSelection const& sel = _view.get_real_output_selection();
	//PVCore::PVHSVColor const* const colors = _pvview_sp->get_output_layer().get_lines_properties().get_buffer();
	//lib_view().get_color_in_output_layer(r);

	for (uint32_t branch = 0 ; branch < NBUCKETS; branch++)
	{
		if (_zt.branch_valid(branch)) {
			const PVRow row = _zt.get_first_elt_of_branch(branch);
			code_b.int_v = branch;
			int32_t x = code_b.s.l;
			int32_t y = code_b.s.r;

			painter->setPen(_view.get_color_in_output_layer(row).toQColor());

			if (sel.get_line_fast(row)) {
				// Draw selection
				painter->setOpacity(1.0);
			}
			else {
				// Draw background
				painter->setOpacity(0.25);
			}
			painter->drawPoint(x+get_real_viewport_rect().x(), y+get_real_viewport_rect().y());
		}
	}
}
