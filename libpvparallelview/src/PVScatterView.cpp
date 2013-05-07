/**
 * \file PVScatterView.cpp
 *
 * Copyright (C) Picviz Labs 2013
 */

#include <pvparallelview/PVScatterView.h>

#include <QApplication>
#include <QGraphicsScene>
#include <QPainter>

#include <pvkernel/widgets/PVGraphicsViewInteractor.h>

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
using PVScatterViewZoomConverter = PVZoomConverterScaledPowerOfTwo<STEPS>;

class PVSelectionSquareInteractor: public PVWidgets::PVGraphicsViewInteractor<PVWidgets::PVGraphicsView>
{

public:
	PVSelectionSquareInteractor(PVWidgets::PVGraphicsView* parent, PVSelectionSquare* selection_square):
		PVWidgets::PVGraphicsViewInteractor<PVWidgets::PVGraphicsView>(parent),
		_selection_square(selection_square)
	{
		assert(selection_square->scene() == parent->get_scene());
	}

	bool keyPressEvent(PVWidgets::PVGraphicsView* view, QKeyEvent* event) override
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

#ifdef PICVIZ_DEVELOPER_MODE
	else if ((event->key() == Qt::Key_B) && (event->modifiers() & Qt::ControlModifier)) {
		PVScatterView::toggle_show_quadtrees();
	}
#endif

		return false;
	}

	bool mousePressEvent(PVWidgets::PVGraphicsView* view, QMouseEvent* event) override
	{
		if (event->button() == Qt::LeftButton) {
			QPointF p = view->map_to_scene(event->pos());
			_selection_square->begin(p.x(), p.y());
			event->accept();
		}
		return false;
	}

	bool mouseReleaseEvent(PVWidgets::PVGraphicsView* view, QMouseEvent* event) override
	{
		if (event->button() == Qt::LeftButton) {
			QPointF p = view->map_to_scene(event->pos());
			_selection_square->end(p.x(), p.y(), true, true);
			event->accept();
		}
		return false;
	}

	bool mouseMoveEvent(PVWidgets::PVGraphicsView* view, QMouseEvent* event) override
	{
		if (event->buttons() == Qt::LeftButton)
		{
			QPointF p = view->map_to_scene(event->pos());
			_selection_square->end(p.x(), p.y());
			event->accept();
		}
		return false;
	}

private:
	PVSelectionSquare* _selection_square;
};

}

bool PVParallelView::PVScatterView::_show_quadtrees = false;

PVParallelView::PVScatterView::PVScatterView(
	const Picviz::PVView_sp &pvview_sp,
	PVZonesManager & zm,
	PVCol const axis_index,
	QWidget* parent /*= nullptr*/
) :
	PVZoomableDrawingAreaWithAxes(parent),
	_view(*pvview_sp),
	_zt(zm.get_zone_tree<PVParallelView::PVZoneTree>(axis_index)),
	_zzt(zm.get_zone_tree<PVParallelView::PVZoomedZoneTree>(axis_index)),
	_view_deleted(false)
{
	//setCursor(Qt::CrossCursor);
	QRectF r(0, 0, (1UL << 32), (1UL << 32));
	set_scene_rect(r);
	get_scene()->setSceneRect(r);

	_selection_square = new PVSelectionSquareScatterView(_zt, this);

	// interactor
	PVWidgets::PVGraphicsViewInteractorBase* zoom_inter = declare_interactor<PVZoomableDrawingAreaInteractorHomothetic>();
	PVWidgets::PVGraphicsViewInteractorBase* selection_square_inter = declare_interactor<PVSelectionSquareInteractor>(_selection_square);
	register_back_all(selection_square_inter);
	register_back_all(zoom_inter);
	install_default_scene_interactor();

	// constraints
	set_constraints(new PVParallelView::PVZoomableDrawingAreaConstraintsHomothetic());

	// decorations
	set_alignment(Qt::AlignLeft | Qt::AlignTop);
	set_horizontal_scrollbar_policy(Qt::ScrollBarAlwaysOn);

	// TODO: register axis name change through the hive
	set_x_legend(pvview_sp->get_axis_name(axis_index));
	set_y_legend(pvview_sp->get_axis_name(axis_index+1));

	set_transformation_anchor(AnchorUnderMouse);

	set_decoration_color(Qt::white);
	set_ticks_per_level(8);

	_zoom_converter = new PVScatterViewZoomConverter<zoom_steps>();
	get_x_axis_zoom().set_zoom_converter(_zoom_converter);
	get_x_axis_zoom().set_range(zoom_min, zoom_extra);
	get_x_axis_zoom().set_default_value(zoom_min);
	get_y_axis_zoom().set_zoom_converter(_zoom_converter);
	get_y_axis_zoom().set_range(zoom_min, zoom_extra);
	get_y_axis_zoom().set_default_value(zoom_min);

	set_zoom_value(PVZoomableDrawingAreaConstraints::X
	               | PVZoomableDrawingAreaConstraints::Y,
	               zoom_min);

	get_scene()->setItemIndexMethod(QGraphicsScene::NoIndex);

	// Request quadtrees creation
	zm.request_zoomed_zone(axis_index);
}

/*****************************************************************************
 * PVParallelView::PVScatterView::~PVScatterView
 *****************************************************************************/

PVParallelView::PVScatterView::~PVScatterView()
{
	if (!_view_deleted) {
		common::get_lib_view(_view)->remove_scatter_view(this);
	}

	delete _zoom_converter;
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
	Picviz::PVSelection const& sel = _view.get_real_output_selection();

	PVZoomedZoneTree::context_t ctxt;

	typedef PVParallelView::PVBCICode<PARALLELVIEW_ZZT_BBITS>bcicode_t;
	static bcicode_t* bcicodes = new bcicode_t[NBUCKETS];

	int rel_zoom = get_y_axis_zoom().get_clamped_relative_value();

	QRectF screen_rect = get_viewport()->rect();
	screen_rect.translate(get_scene_left_margin(), get_scene_top_margin());
	QRectF screen_rect_s = map_to_scene(screen_rect);
	QRectF view_rect = get_scene_rect().intersected(screen_rect_s);

	uint64_t y1_min = view_rect.x();
	uint64_t y1_max = view_rect.x()+view_rect.width();
	uint64_t y2_min = view_rect.y();
	uint64_t y2_max = view_rect.y()+view_rect.height();

	double alpha = 0.5 * _zoom_converter->zoom_to_scale_decimal(rel_zoom);

	size_t bci_count = _zzt.browse_bci_by_y1_y2(
		ctxt,
		y1_min,
		y1_max,
		y2_min,
		y2_max,
		rel_zoom/zoom_steps,
		alpha,
		bcicodes
	);

	painter->setPen(Qt::white);
	painter->setOpacity(1.0);

	for (uint32_t i = 0; i < bci_count; ++i) {
		bcicode_t code_bci = bcicodes[i];
		painter->setPen(_view.get_color_in_output_layer(code_bci.s.idx).toQColor());
		painter->drawPoint(code_bci.s.l + get_scene_left_margin(), code_bci.s.r+ get_scene_top_margin());
		//painter->drawEllipse(code_bci.s.l + get_scene_left_margin(), code_bci.s.r + get_scene_top_margin(), 10, 10);
	}

	if (_show_quadtrees) {
		PVParallelView::PVBCode code_b;
		for (uint32_t branch = 0 ; branch < NBUCKETS; branch++)
		{
			if (_zt.branch_valid(branch)) {
				const PVRow row = _zt.get_first_elt_of_branch(branch);
				code_b.int_v = branch;
				const double x_scene = ((uint32_t)code_b.s.l) << (32-PARALLELVIEW_ZT_BBITS);
				const double y_scene = ((uint32_t)code_b.s.r) << (32-PARALLELVIEW_ZT_BBITS);

				const double x_rect_scene = ((uint32_t)((code_b.s.l+1) << (32-PARALLELVIEW_ZT_BBITS))) - 1;
				const double y_rect_scene = ((uint32_t)((code_b.s.r+1) << (32-PARALLELVIEW_ZT_BBITS))) - 1;

				QPointF view_point = map_from_scene(QPointF(x_scene, y_scene));
				QPointF view_point_rect = map_from_scene(QPointF(x_rect_scene, y_rect_scene));

				painter->setPen(_view.get_color_in_output_layer(row).toQColor());

				if (sel.get_line_fast(row)) {
					// Draw selection
					painter->setOpacity(1.0);
				}
				else {
					// Draw background
					painter->setOpacity(0.25);
				}
				painter->drawRect(QRectF(view_point, view_point_rect));
			}
		}
	}
}
