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
#include <pvparallelview/PVZoomableDrawingAreaInteractorHomothetic.h>
#include <pvparallelview/PVZoomableDrawingAreaConstraintsHomothetic.h>

namespace PVParallelView
{

template <int STEPS>
class PVScatterViewZoomConverter : public PVZoomConverterScaledPowerOfTwo<STEPS>
{
public:
	PVScatterViewZoomConverter(const qreal s) :
		_scale_factor(s)
	{}

	int scale_to_zoom(const qreal value) const override
	{
		return PVZoomConverterScaledPowerOfTwo<STEPS>::scale_to_zoom(value / _scale_factor);
	}

	qreal zoom_to_scale(const int value) const override
	{
		return PVZoomConverterScaledPowerOfTwo<STEPS>::zoom_to_scale(value) * _scale_factor;
	}

private:
	qreal _scale_factor;
};

class PVScatterViewInteractor : public PVZoomableDrawingAreaInteractorHomothetic
{

public:
	PVScatterViewInteractor(PVWidgets::PVGraphicsView* parent, Picviz::PVView& view, const PVZoneTree &zt) :
		PVZoomableDrawingAreaInteractorHomothetic(parent),
		_view(view),
		_scatter_view(static_cast<PVScatterView*>(parent)),
		_selection_square(new PVSelectionSquareScatterView(view, zt, _scatter_view->get_scene()))
	{}

	bool keyPressEvent(PVZoomableDrawingArea* zda, QKeyEvent* event) override
	{
		PVZoomableDrawingAreaInteractorHomothetic::keyPressEvent(zda, event);

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

	bool mousePressEvent(PVZoomableDrawingArea* obj, QMouseEvent* event) override
	{
		PVZoomableDrawingAreaInteractorHomothetic::mousePressEvent(obj, event);

		if (event->button() == Qt::LeftButton) {
			QPointF p = obj->map_to_scene(event->pos());
			_selection_square->begin(p.x(), p.y());
			event->accept();
		}
		return false;
	}

	bool mouseReleaseEvent(PVZoomableDrawingArea* obj, QMouseEvent* event) override
	{
		PVZoomableDrawingAreaInteractorHomothetic::mouseReleaseEvent(obj, event);

		if (event->button() == Qt::LeftButton) {
			QPointF p = obj->map_to_scene(event->pos());
			_selection_square->end(p.x(), p.y(), true, true);
			event->accept();
		}
		return false;
	}

	bool mouseMoveEvent(PVZoomableDrawingArea* obj, QMouseEvent* event) override
	{
		PVZoomableDrawingAreaInteractorHomothetic::mouseMoveEvent(obj, event);

		if (event->buttons() == Qt::LeftButton)
		{
			QPointF p = obj->map_to_scene(event->pos());
			_selection_square->end(p.x(), p.y());
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
	QRectF r(0, -(1L << 32), (1L << 32), (1L << 32));
	set_scene_rect(r);
	get_scene()->setSceneRect(r);
	//get_scene()->setSceneRect(0, 0, 1024, 1024);

	// interactor
	PVWidgets::PVGraphicsViewInteractorBase* inter = declare_interactor<PVScatterViewInteractor>(_view, _zt);
	register_front_all(inter);
	register_front_one(QEvent::Resize, inter);
	register_front_one(QEvent::KeyPress, inter);
	install_default_scene_interactor();

	// constraints
	set_constraints(new PVParallelView::PVZoomableDrawingAreaConstraintsHomothetic());

	// decorations
	set_alignment(Qt::AlignLeft | Qt::AlignTop);
	#if 0
		set_horizontal_scrollbar_policy(Qt::ScrollBarAlwaysOff);
	#else
		set_horizontal_scrollbar_policy(Qt::ScrollBarAlwaysOn);
	#endif

	//set_x_legend("occurrence count");
	//set_y_legend(pvview_sp->get_axis_name(axis_index));
	set_decoration_color(Qt::white);
	set_ticks_per_level(8);

	_zoom_converter = new PVScatterViewZoomConverter<zoom_steps>(r.height() / r.width());
	get_x_axis_zoom().set_range(-110, 30);
	get_x_axis_zoom().set_default_value(-110);
	get_x_axis_zoom().set_zoom_converter(_zoom_converter);
	get_y_axis_zoom().set_range(-110, 30);
	get_y_axis_zoom().set_default_value(-110);
	get_y_axis_zoom().set_zoom_converter(_zoom_converter);
	set_zoom_value(PVZoomableDrawingAreaConstraints::X | PVZoomableDrawingAreaConstraints::Y, -110);
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

	//painter->fillRect(rect, QColor::fromRgbF(0.1, 0.1, 0.1, 1.0));
	//PVZoomableDrawingAreaWithAxes::drawBackground(painter, rect);
	recompute_decorations(painter, rect);

	draw_points(painter, rect);

	painter->setOpacity(1.0);
	painter->setPen(QPen(Qt::white));
	draw_decorations(painter, rect);
}

/*****************************************************************************
 * PVParallelView::PVScatterView::draw_points
 *****************************************************************************/
void PVParallelView::PVScatterView::draw_points(QPainter* painter, const QRectF& rect)
{
	PVParallelView::PVBCode code_b;

	Picviz::PVSelection const& sel = _view.get_real_output_selection();
	//PVCore::PVHSVColor const* const colors = _pvview_sp->get_output_layer().get_lines_properties().get_buffer();
	//lib_view().get_color_in_output_layer(r);

	qreal ref_left = get_scene_left_margin();
	qreal ref_bottom = get_scene_top_margin() + get_y_axis_length();

	for (uint32_t branch = 0 ; branch < NBUCKETS; branch++)
	{
		if (_zt.branch_valid(branch)) {
			const PVRow row = _zt.get_first_elt_of_branch(branch);
			code_b.int_v = branch;
			int32_t x = ref_left + code_b.s.l;
			int32_t y = ref_bottom - code_b.s.r;

			if (!get_real_viewport_rect().contains(x, y)) {
				continue;
			}

			painter->setPen(_view.get_color_in_output_layer(row).toQColor());

			if (sel.get_line_fast(row)) {
				// Draw selection
				painter->setOpacity(1.0);
			}
			else {
				// Draw background
				painter->setOpacity(0.25);
			}
			painter->drawPoint(x, y);
		}
	}
}
