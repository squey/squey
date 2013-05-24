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
#include <pvparallelview/PVZoneRenderingScatter.h>
#include <pvparallelview/PVSelectionRectangleInteractor.h>

namespace PVParallelView
{

template <int STEPS>
using PVScatterViewZoomConverter = PVZoomConverterScaledPowerOfTwo<STEPS>;

class PVScatterViewInteractor: public PVWidgets::PVGraphicsViewInteractor<PVScatterView>
{
public:
	PVScatterViewInteractor(PVWidgets::PVGraphicsView* parent = nullptr) :
		PVGraphicsViewInteractor<PVScatterView>(parent)
	{ }

public:
	bool resizeEvent(PVScatterView* view, QResizeEvent*) override
	{
		view->update_all_async();
		return false;
	}
};

}

bool PVParallelView::PVScatterView::_show_quadtrees = false;

PVParallelView::PVScatterView::PVScatterView(
	const Picviz::PVView_sp &pvview_sp,
	PVZonesManager const& zm,
	PVCol const zone_index,
	PVZonesProcessor& zp_bg,
	PVZonesProcessor& zp_sel,
	QWidget* parent /*= nullptr*/
) :
	PVZoomableDrawingAreaWithAxes(parent),
	_view(*pvview_sp),
	_images_manager(zone_index, zp_bg, zp_sel, zm, pvview_sp->output_layer.get_lines_properties().get_buffer(), pvview_sp->get_real_output_selection()),
	_view_deleted(false)
{
	//set_gl_viewport();

	set_x_axis_inverted(true);

	setCursor(Qt::CrossCursor);
	QRectF r(0, 0, (1UL << 32), (1UL << 32));
	set_scene_rect(r);
	get_scene()->setSceneRect(r);

	_selection_square = new PVSelectionSquareScatterView(this);

	// interactor
	PVWidgets::PVGraphicsViewInteractorBase* zoom_inter = declare_interactor<PVZoomableDrawingAreaInteractorHomothetic>();
	PVWidgets::PVGraphicsViewInteractorBase* selection_square_inter = declare_interactor<PVSelectionRectangleInteractor>(_selection_square);
	PVWidgets::PVGraphicsViewInteractorBase* scatter_inter = declare_interactor<PVScatterViewInteractor>();
	register_back_all(selection_square_inter);
	register_back_all(zoom_inter);
	register_back_all(scatter_inter);
	install_default_scene_interactor();

	// constraints
	set_constraints(new PVParallelView::PVZoomableDrawingAreaConstraintsHomothetic());

	// decorations
	set_alignment(Qt::AlignLeft | Qt::AlignTop);
	set_horizontal_scrollbar_policy(Qt::ScrollBarAlwaysOn);

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

	set_scatter_view_zone(zone_index);

	get_scene()->setItemIndexMethod(QGraphicsScene::NoIndex);

	connect(this, SIGNAL(zoom_has_changed(int)), this, SLOT(do_zoom_change(int)));
	connect(this, SIGNAL(pan_has_changed()), this, SLOT(do_pan_change()));
	connect(get_vertical_scrollbar(), SIGNAL(valueChanged(qint64)), this, SLOT(do_pan_change()));
	connect(get_horizontal_scrollbar(), SIGNAL(valueChanged(qint64)), this, SLOT(do_pan_change()));

	get_images_manager().set_img_update_receiver(this);
}

/*****************************************************************************
 * PVParallelView::PVScatterView::~PVScatterView
 *****************************************************************************/

PVParallelView::PVScatterView::~PVScatterView()
{
	get_images_manager().cancel_all_and_wait();

	if (!_view_deleted) {
		common::get_lib_view(_view)->remove_scatter_view(this);
	}

	delete _zoom_converter;
}

PVParallelView::PVZoneTree const& PVParallelView::PVScatterView::get_zone_tree() const
{
	return get_zones_manager().get_zone_tree<PVParallelView::PVZoneTree>(get_zone_index());
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
	QMetaObject::invokeMethod(this, "update_sel", Qt::QueuedConnection);
}

/*****************************************************************************
 * PVParallelView::PVScatterView::update_all_async
 *****************************************************************************/
void PVParallelView::PVScatterView::update_all_async()
{
	QMetaObject::invokeMethod(this, "update_all", Qt::QueuedConnection);
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
		get_viewport()->update();
#endif
}

/*****************************************************************************
 * PVParallelView::PVScatterView::do_zoom_change
 *****************************************************************************/
void PVParallelView::PVScatterView::do_zoom_change(int /*axes*/)
{
	do_update_all();
}

/*****************************************************************************
 * PVParallelView::PVScatterView::do_pan_change
 *****************************************************************************/
void PVParallelView::PVScatterView::do_pan_change()
{
	do_update_all();
}

/*****************************************************************************
 * PVParallelView::PVScatterView::update_all
 *****************************************************************************/
void PVParallelView::PVScatterView::update_all()
{
	get_images_manager().process_all();
}

/*****************************************************************************
 * PVParallelView::PVScatterView::update_sel
 *****************************************************************************/
void PVParallelView::PVScatterView::update_sel()
{
	get_images_manager().process_sel();
}

void PVParallelView::PVScatterView::update_img_bg(PVZoneRendering_p zr, int /*zone*/)
{
	assert(QThread::currentThread() == thread());
	if (zr->should_cancel()) {
		return;
	}

	_image_bg.swap(get_images_manager().get_image_all(), _last_image_scene, _last_image_margined_viewport);
	get_viewport()->update();
}

void PVParallelView::PVScatterView::update_img_sel(PVZoneRendering_p zr, int /*zone*/)
{
	assert(QThread::currentThread() == thread());
	if (zr->should_cancel()) {
		return;
	}

	_image_sel.swap(get_images_manager().get_image_sel(), _last_image_scene, _last_image_margined_viewport);
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

	_last_image_margined_viewport = QRectF(0.0, 0.0, get_x_axis_length(), get_y_axis_length());
	_last_image_scene = QRectF(QPointF(y1_min, y2_min), QPointF(y1_max, y2_max));

	get_images_manager().change_and_process_view(y1_min, y1_max, y2_min, y2_max, zoom, alpha);
	get_viewport()->update();
}

bool PVParallelView::PVScatterView::update_zones()
{
	PVCol new_zone = lib_view().get_axes_combination().get_index_by_id(_axis_id);
	if (new_zone == PVCOL_INVALID_VALUE) {
		if (get_zone_index() > get_zones_manager().get_number_of_managed_zones()) {
			// Just delete this view as it can't be replaced by anything...
			return false;
		}

		new_zone = get_zone_index();
	}

	set_scatter_view_zone(new_zone);

	return true;
}

void PVParallelView::PVScatterView::set_scatter_view_zone(PVZoneID const zid)
{
	_axis_id = lib_view().get_axes_combination().get_axes_comb_id(zid);
	const uint32_t *y1_plotted, *y2_plotted;
	get_zones_manager().get_zone_plotteds(zid, &y1_plotted, &y2_plotted);
	get_images_manager().set_zone(zid);
	_selection_square->set_plotteds(y1_plotted, y2_plotted, get_zones_manager().get_number_rows());

	// TODO: register axis name change through the hive
	set_x_legend(lib_view().get_axis_name(zid));
	set_y_legend(lib_view().get_axis_name(zid+1));
}

/*****************************************************************************
 * PVParallelView::PVScatterView::drawBackground
 *****************************************************************************/
void PVParallelView::PVScatterView::drawBackground(QPainter* painter, const QRectF& rect)
{
	painter->fillRect(rect, QColor::fromRgbF(0.1, 0.1, 0.1, 1.0));

	painter->save();

	const QRect margined_viewport = QRect(0, 0, get_x_axis_length(), get_y_axis_length());
	painter->setClipRegion(margined_viewport, Qt::IntersectClip);

	// Background
	painter->setOpacity(0.25);
	_image_bg.draw(this, painter);

	// Selection
	painter->setOpacity(1);
	_image_sel.draw(this, painter);

#ifdef PICVIZ_DEVELOPER_MODE
	if (_show_quadtrees) {
		painter->setPen(Qt::white);
		painter->setOpacity(1.0);
		const Picviz::PVSelection& sel = _view.get_real_output_selection();
		PVParallelView::PVBCode code_b;
		PVParallelView::PVZoneTree const& zt = get_zone_tree();
		for (uint32_t branch = 0 ; branch < NBUCKETS; branch++)
		{
			if (zt.branch_valid(branch)) {
				const PVRow row = zt.get_first_elt_of_branch(branch);
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

	painter->restore();

	draw_decorations(painter, rect);
}

void PVParallelView::PVScatterView::set_enabled(bool en)
{
	setEnabled(en);
	if (!en) {
		get_images_manager().cancel_all_and_wait();
	}
}

////
// PVParallelView::PVScatterView::RenderedImage implementation
////

void PVParallelView::PVScatterView::RenderedImage::swap(QImage const& img, QRectF const& scene_rect, QRectF const& viewport_rect)
{
	_scene_rect = scene_rect;
	_viewport_rect = viewport_rect;
	_img = img.copy(_viewport_rect.toAlignedRect()).mirrored(true, false);
}

void PVParallelView::PVScatterView::RenderedImage::draw(PVGraphicsView* view, QPainter* painter)
{
	const QRectF target_sel = view->map_margined_from_scene(_scene_rect);
	painter->drawImage(target_sel, _img);
}
