/**
 * \file PVScatterView.cpp
 *
 * Copyright (C) Picviz Labs 2013
 */

#include <pvparallelview/PVScatterView.h>

#include <QApplication>
#include <QGraphicsScene>
#include <QPainter>
#include <QScrollBar64>

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
#include <pvparallelview/PVZoomConverterPowerOfTwo.h>
#include <pvparallelview/PVSelectionRectangleInteractor.h>

namespace PVParallelView
{

template <int STEPS>
using PVScatterViewZoomConverter = PVZoomConverterScaledPowerOfTwo<STEPS>;

}

bool PVParallelView::PVScatterView::_show_quadtrees = false;

PVParallelView::PVScatterView::PVScatterView(
	const Picviz::PVView_sp &pvview_sp,
	PVZonesManager & zm,
	PVCol const axis_index,
	QWidget* parent /*= nullptr*/
) :
	PVZoomableDrawingAreaWithAxes(parent),
	_images_manager(zm.get_zone_tree<PVParallelView::PVZoomedZoneTree>(axis_index), pvview_sp->output_layer.get_lines_properties().get_buffer(), _view.get_real_output_selection()),
	_view(*pvview_sp),
	_zt(zm.get_zone_tree<PVParallelView::PVZoneTree>(axis_index)),
	_view_deleted(false)
{
	setCursor(Qt::CrossCursor);
	QRectF r(0, 0, (1UL << 32), (1UL << 32));
	set_scene_rect(r);
	get_scene()->setSceneRect(r);

	const PVRow nrows = zm.get_number_rows();

	const uint32_t* y1_plotted = Picviz::PVPlotted::get_plotted_col_addr(
		zm.get_uint_plotted(),
		nrows,
		axis_index
	);

	const uint32_t* y2_plotted = Picviz::PVPlotted::get_plotted_col_addr(
		zm.get_uint_plotted(),
		nrows,
		axis_index+1
	);

	_selection_square = new PVSelectionSquareScatterView(y1_plotted, y2_plotted, nrows, this);

	// interactor
	PVWidgets::PVGraphicsViewInteractorBase* zoom_inter = declare_interactor<PVZoomableDrawingAreaInteractorHomothetic>();
	PVWidgets::PVGraphicsViewInteractorBase* selection_square_inter = declare_interactor<PVSelectionRectangleInteractor>(_selection_square);
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

	//_zoom_converter = new PVZoomConverterPowerOfTwo();
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

	_update_all_timer.setInterval(render_timer_ms);
	_update_all_timer.setSingleShot(true);
	connect(&_update_all_timer, SIGNAL(timeout()), this, SLOT(do_update_all()));

	connect(this, SIGNAL(zoom_has_changed(int)), this, SLOT(do_zoom_change(int)));
	connect(this, SIGNAL(pan_has_changed()), this, SLOT(do_pan_change()));
	connect(get_vertical_scrollbar(), SIGNAL(valueChanged(qint64)), this, SLOT(do_pan_change()));

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
	QMetaObject::invokeMethod(this, "update_all", Qt::QueuedConnection);
}

/*****************************************************************************
 * PVParallelView::PVScatterView::update_all_async
 *****************************************************************************/
void PVParallelView::PVScatterView::update_all_async()
{
	QMetaObject::invokeMethod(this, "update_sel", Qt::QueuedConnection);
}

/*****************************************************************************
 * PVParallelView::PVScatterView::keyPressEvent
 *****************************************************************************/
void PVParallelView::PVScatterView::keyPressEvent(QKeyEvent* event)
{
		PVZoomableDrawingAreaWithAxes::keyPressEvent(event);
#ifdef PICVIZ_DEVELOPER_MODE
		if ((event->key() == Qt::Key_B) && (event->modifiers() & Qt::ControlModifier)) {
			PVScatterView::toggle_show_quadtrees();
		}
		update();
#endif
}

/*****************************************************************************
 * PVParallelView::PVScatterView::do_zoom_change
 *****************************************************************************/
void PVParallelView::PVScatterView::do_zoom_change(int /*axes*/)
{
	_update_all_timer.start();
}

/*****************************************************************************
 * PVParallelView::PVScatterView::do_pan_change
 *****************************************************************************/
void PVParallelView::PVScatterView::do_pan_change()
{
	_update_all_timer.start();
}

/*****************************************************************************
 * PVParallelView::PVScatterView::update_all
 *****************************************************************************/
void PVParallelView::PVScatterView::update_all()
{
	get_images_manager().process_all();
	get_viewport()->update();
}

/*****************************************************************************
 * PVParallelView::PVScatterView::update_sel
 *****************************************************************************/
void PVParallelView::PVScatterView::update_sel()
{
	get_images_manager().process_sel();
	get_viewport()->update();
}

/*****************************************************************************
 * PVParallelView::PVScatterView::do_update_all
 *****************************************************************************/
void PVParallelView::PVScatterView::do_update_all()
{
	QRectF view_rect = get_scene_rect().intersected(map_to_scene(get_margined_viewport_rect()));

	uint64_t y1_min = view_rect.x();
	uint64_t y1_max = view_rect.x()+view_rect.width();
	uint64_t y2_min = view_rect.y();
	uint64_t y2_max = view_rect.y()+view_rect.height();
	int64_t zoom = get_y_axis_zoom().get_clamped_relative_value();
	double alpha = 0.5 * _zoom_converter->zoom_to_scale_decimal(zoom);
	zoom = (zoom / zoom_steps) +1;

	get_images_manager().change_and_process_view(y1_min, y1_max, y2_min, y2_max, zoom, alpha);
	_last_image_margined_viewport = QRectF(0.0, 0.0, get_margined_viewport_width(), get_margined_viewport_height());

	get_viewport()->update();
}

/*****************************************************************************
 * PVParallelView::PVScatterView::drawBackground
 *****************************************************************************/
void PVParallelView::PVScatterView::drawBackground(QPainter* painter, const QRectF& rect)
{
	painter->fillRect(rect, QColor::fromRgbF(0.1, 0.1, 0.1, 1.0));

	const QRectF img_scene(QPointF(get_images_manager().last_y1_min(), get_images_manager().last_y2_min()),
			               QPointF(get_images_manager().last_y1_max(), get_images_manager().last_y2_max()));
	const QRect margined_viewport = QRect(0, 0, get_margined_viewport_width(), get_margined_viewport_height());
	const QRectF target = map_margined_from_scene(img_scene);

	painter->save();
	painter->setClipRegion(margined_viewport, Qt::IntersectClip);
	painter->setOpacity(1.0);
	painter->drawImage(target, get_images_manager().get_image_all(), _last_image_margined_viewport);

	painter->restore();

#ifdef PICVIZ_DEVELOPER_MODE
	if (_show_quadtrees) {
		painter->setPen(Qt::white);
		painter->setOpacity(1.0);
		const Picviz::PVSelection& sel = _view.get_real_output_selection();
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

				QPointF view_point = map_margined_from_scene(QPointF(x_scene, y_scene));
				QPointF view_point_rect = map_margined_from_scene(QPointF(x_rect_scene, y_rect_scene));

				painter->setPen(_view.get_color_in_output_layer(row).toQColor());
				painter->setOpacity(sel.get_line_fast(row) ? 1.0 : 0.25);
				painter->drawRect(QRectF(view_point, view_point_rect));
			}
		}
	}
#endif

	painter->setOpacity(1.0);
	painter->setPen(QPen(Qt::white));
	draw_decorations(painter, rect);
}
