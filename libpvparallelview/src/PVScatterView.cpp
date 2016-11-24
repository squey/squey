/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvparallelview/PVBCode.h>
#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVScatterView.h>
#include <pvparallelview/PVScatterViewInteractor.h>
#include <pvparallelview/PVScatterViewParamsWidget.h>
#include <pvparallelview/PVScatterViewSelectionRectangle.h>
#include <pvparallelview/PVSelectionRectangleInteractor.h>
#include <pvparallelview/PVZoneRenderingScatter.h>
#include <pvparallelview/PVZoneTree.h>
#include <pvparallelview/PVZoomConverterPowerOfTwo.h>
#include <pvparallelview/PVZoomableDrawingAreaConstraintsHomothetic.h>
#include <pvparallelview/PVZoomableDrawingAreaInteractor.h>
#include <pvparallelview/PVZoomableDrawingAreaInteractorHomothetic.h>

#include <inendi/PVView.h>

#include <pvkernel/widgets/PVHelpWidget.h>
#include <pvkernel/rush/PVNraw.h>
#include <pvkernel/widgets/PVGraphicsViewInteractor.h>

#include <QAction>
#include <QActionGroup>
#include <QApplication>
#include <QButtonGroup>
#include <QGraphicsScene>
#include <QPainter>
#include <QScrollBar64>
#include <QThread>
#include <QToolBar>
#include <QToolButton>

namespace PVParallelView
{

template <int STEPS>
using PVScatterViewZoomConverter = PVZoomConverterScaledPowerOfTwo<STEPS>;
} // namespace PVParallelView

bool PVParallelView::PVScatterView::_show_quadtrees = false;

PVParallelView::PVScatterView::PVScatterView(Inendi::PVView& pvview_sp,
                                             PVZonesManager const& zm,
                                             PVCol const zone_index,
                                             PVZonesProcessor& zp_bg,
                                             PVZonesProcessor& zp_sel,
                                             QWidget* parent /*= nullptr*/
                                             )
    : PVZoomableDrawingAreaWithAxes(parent)
    , _view(pvview_sp)
    , _images_manager(zone_index,
                      zp_bg,
                      zp_sel,
                      zm,
                      pvview_sp.get_output_layer_color_buffer(),
                      pvview_sp.get_real_output_selection())
    , _view_deleted(false)
    , _show_bg(true)
{
	set_gl_viewport();

	QRectF r(0, 0, (1UL << 32), (1UL << 32));
	set_scene_rect(r);
	get_scene()->setSceneRect(r);

	/* zoom converter
	 */
	_zoom_converter = new PVScatterViewZoomConverter<zoom_steps>();
	get_x_axis_zoom().set_zoom_converter(_zoom_converter);
	get_y_axis_zoom().set_zoom_converter(_zoom_converter);

	_sel_rect = new PVScatterViewSelectionRectangle(this);

	/* interactor
	 */
	_sel_rect_interactor = declare_interactor<PVSelectionRectangleInteractor>(_sel_rect);
	register_front_all(_sel_rect_interactor);

	_h_interactor = declare_interactor<PVZoomableDrawingAreaInteractorHomothetic>();
	register_front_all(_h_interactor);

	_sv_interactor = declare_interactor<PVScatterViewInteractor>();
	register_front_all(_sv_interactor);

	install_default_scene_interactor();

	// need to move them to front to allow view pan before sel rect move
	register_front_one(QEvent::MouseButtonPress, _h_interactor);
	register_front_one(QEvent::MouseButtonRelease, _h_interactor);
	register_front_one(QEvent::MouseMove, _h_interactor);

	/* constraints
	 */
	set_constraints(new PVParallelView::PVZoomableDrawingAreaConstraintsHomothetic());

	/* axis zoom
	 */
	get_x_axis_zoom().set_range(zoom_min, zoom_extra);
	get_x_axis_zoom().set_default_value(zoom_min);
	get_y_axis_zoom().set_range(zoom_min, zoom_extra);
	get_y_axis_zoom().set_default_value(zoom_min);

	set_zoom_value(PVZoomableDrawingAreaConstraints::X | PVZoomableDrawingAreaConstraints::Y,
	               zoom_min);

	// decorations
	set_alignment(Qt::AlignLeft | Qt::AlignTop);
	set_horizontal_scrollbar_policy(Qt::ScrollBarAlwaysOn);

	set_transformation_anchor(AnchorUnderMouse);

	set_decoration_color(Qt::white);
	set_ticks_per_level(8);

	set_scatter_view_zone(zone_index);

	get_scene()->setItemIndexMethod(QGraphicsScene::NoIndex);

	connect(this, SIGNAL(zoom_has_changed(int)), this, SLOT(do_zoom_change(int)));
	connect(this, SIGNAL(pan_has_changed()), this, SLOT(do_pan_change()));
	connect(get_vertical_scrollbar(), SIGNAL(valueChanged(qint64)), this, SLOT(do_pan_change()));
	connect(get_horizontal_scrollbar(), SIGNAL(valueChanged(qint64)), this, SLOT(do_pan_change()));

	_params_widget = new PVScatterViewParamsWidget(this);
	_params_widget->setAutoFillBackground(true);
	_params_widget->adjustSize();
	set_params_widget_position();

	get_images_manager().set_img_update_receiver(this);

	_help_widget = new PVWidgets::PVHelpWidget(this);
	_help_widget->hide();

	_help_widget->initTextFromFile("scatter view's help", ":help-style");
	_help_widget->addTextFromFile(":help-selection");
	_help_widget->addTextFromFile(":help-layers");
	_help_widget->newColumn();
	_help_widget->addTextFromFile(":help-lines");
	_help_widget->addTextFromFile(":help-application");

	_help_widget->newTable();
	_help_widget->addTextFromFile(":help-mouse-view");
	_help_widget->newColumn();
	_help_widget->addTextFromFile(":help-sel-rect-full");

	_help_widget->newTable();
	_help_widget->addTextFromFile(":help-mouse-scatter-view");
	_help_widget->finalizeText();

	// Register view for unselected & zombie events toggle
	pvview_sp._toggle_unselected_zombie_visibility.connect(
	    sigc::mem_fun(this, &PVParallelView::PVScatterView::toggle_unselected_zombie_visibility));

	_sel_rect->set_default_cursor(Qt::CrossCursor);
	set_viewport_cursor(Qt::CrossCursor);
	set_background_color(common::color_view_bg());

	_sel_rect->set_x_range(0, UINT32_MAX);
	_sel_rect->set_y_range(0, UINT32_MAX);
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
	delete get_constraints();
	delete _h_interactor;
	delete _sv_interactor;
	delete _sel_rect_interactor;
	delete _sel_rect;
}

PVParallelView::PVZoneTree const& PVParallelView::PVScatterView::get_zone_tree() const
{
	return get_zones_manager().get_zone_tree(get_zone_index());
}

/*****************************************************************************
 * PVParallelView::PVScatterView::about_to_be_deleted
 *****************************************************************************/
void PVParallelView::PVScatterView::about_to_be_deleted()
{
	_view_deleted = true;
}

/*****************************************************************************
 * PVParallelView::PVScatterView::set_params_widget_position
 *****************************************************************************/
void PVParallelView::PVScatterView::set_params_widget_position()
{
	QPoint pos = QPoint(get_viewport()->width() - 4, 4);
	pos -= QPoint(_params_widget->width(), 0);
	_params_widget->move(pos);
	_params_widget->raise();
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
#ifdef INENDI_DEVELOPER_MODE
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

	_image_bg.swap(get_images_manager().get_image_all(), _last_image_margined_viewport,
	               _last_image_mv2s);
	get_viewport()->update();
}

void PVParallelView::PVScatterView::update_img_sel(PVZoneRendering_p zr, int /*zone*/)
{
	assert(QThread::currentThread() == thread());
	if (zr->should_cancel()) {
		return;
	}

	_image_sel.swap(get_images_manager().get_image_sel(), _last_image_margined_viewport,
	                _last_image_mv2s);
	get_viewport()->update();
}

/******************************************************************************
 * PVParallelView::PVScatterView::toggle_unselected_zombie_visibility
 *****************************************************************************/

void PVParallelView::PVScatterView::toggle_unselected_zombie_visibility()
{
	_show_bg = _view.are_view_unselected_zombie_visible();

	get_viewport()->update();
}

/*****************************************************************************
 * PVParallelView::PVScatterView::do_update_all
 *****************************************************************************/
void PVParallelView::PVScatterView::do_update_all()
{
	QRectF view_rect = get_scene_rect().intersected(map_to_scene(get_margined_viewport_rect()));

	uint64_t y1_min, y1_max, y2_min, y2_max;
	int64_t zoom;
	double alpha;

	_sel_rect->set_handles_scale(1. / get_transform().m11(), 1. / get_transform().m22());

	if (get_y_axis_zoom().get_clamped_value() < zoom_min) {
		get_viewport()->update();
		return;
	}

	// Hack to revert plotting inversion
	y1_min = PVCore::invert_plotting_value(view_rect.x() + view_rect.width());
	y1_max = PVCore::invert_plotting_value(view_rect.x());
	y2_min = view_rect.y();
	y2_max = view_rect.y() + view_rect.height();

	zoom = (get_y_axis_zoom().get_clamped_value() - zoom_min);
	alpha = 0.5 * _zoom_converter->zoom_to_scale_decimal(zoom);
	zoom = (zoom / zoom_steps) + 1;

	_last_image_margined_viewport = QRectF(0.0, 0.0, get_x_axis_length(), get_y_axis_length());

	_last_image_mv2s = get_transform_from_margined_viewport() * get_transform_to_scene();

	// PVLOG_INFO("y1_min: %u / y2_min: %u\n", y1_min, y2_min);
	get_images_manager().change_and_process_view(y1_min, y1_max, y2_min, y2_max, zoom, alpha);

	get_viewport()->update();
}

bool PVParallelView::PVScatterView::update_zones()
{
	PVCol new_zone = lib_view().get_axes_combination().get_first_comb_col(_nraw_col);
	if (new_zone == PVCOL_INVALID_VALUE) {
		// The left axis of the view have been remove, close the scatter view
		return false;
	} else if (new_zone == lib_view().get_column_count() - 1) {
		// It is the right most axes, we can't create a scatter with another axes. Close the scatter
		// view.
		return false;
	}
	// TODO : We should also close if the right axes is removed.

	set_scatter_view_zone(new_zone);

	return true;
}

void PVParallelView::PVScatterView::set_scatter_view_zone(PVZoneID const zid)
{
	_nraw_col = lib_view().get_axes_combination().get_nraw_axis(zid);
	get_images_manager().set_zone(zid);
	PVZoneProcessing zp = get_zones_manager().get_zone_processing(zid);
	_sel_rect->set_plotteds(zp.plotted_a, zp.plotted_b, zp.size);

	// TODO: register axis name change through the hive
	set_x_legend(lib_view().get_axis_name(zid));
	set_y_legend(lib_view().get_axis_name(zid + 1));
}

/*****************************************************************************
 * PVParallelView::PVScatterView::drawBackground
 *****************************************************************************/
void PVParallelView::PVScatterView::drawBackground(QPainter* painter, const QRectF& rect)
{
	painter->save();

	const QRect margined_viewport = QRect(-1, -1, get_x_axis_length() + 4, get_y_axis_length() + 2);
	painter->setClipRegion(margined_viewport, Qt::IntersectClip);

	if (show_bg()) {
		// Background
		painter->setOpacity(0.25);
		_image_bg.draw(this, painter);
	}

	// Selection
	painter->setOpacity(1);
	_image_sel.draw(this, painter);

#ifdef INENDI_DEVELOPER_MODE
	if (_show_quadtrees) {
		painter->setPen(QPen(Qt::white, 0));
		painter->setOpacity(1.0);
		const Inendi::PVSelection& sel = _view.get_real_output_selection();
		PVParallelView::PVBCode code_b;
		PVParallelView::PVZoneTree const& zt = get_zone_tree();
		for (uint32_t branch = 0; branch < NBUCKETS; branch++) {
			if (zt.branch_valid(branch)) {
				const PVRow row = zt.get_first_elt_of_branch(branch);
				code_b.int_v = branch;
				const double x_scene = ((uint32_t)code_b.s.l) << (32 - PARALLELVIEW_ZT_BBITS);
				const double y_scene = ((uint32_t)code_b.s.r) << (32 - PARALLELVIEW_ZT_BBITS);

				const double x_rect_scene =
				    ((uint32_t)((code_b.s.l + 1) << (32 - PARALLELVIEW_ZT_BBITS))) - 1;
				const double y_rect_scene =
				    ((uint32_t)((code_b.s.r + 1) << (32 - PARALLELVIEW_ZT_BBITS))) - 1;

				QPointF view_point = map_margined_from_scene(QPointF(x_scene, y_scene));
				QPointF view_point_rect =
				    map_margined_from_scene(QPointF(x_rect_scene, y_rect_scene));

				painter->setPen(QPen(_view.get_color_in_output_layer(row).toQColor(), 0));
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

QString PVParallelView::PVScatterView::get_x_value_at(const qint64 value)
{
	PVParallelView::PVZoneTree const& zt = get_zone_tree();

	PVParallelView::PVBCode x_b_code;
	x_b_code.int_v = 0;
	// Look for all "upper" left value
	qint64 uvalue = std::numeric_limits<uint32_t>::max() - value;
	for (uint32_t v = (uvalue >> (32 - PARALLELVIEW_ZT_BBITS)); v < ((1 << PARALLELVIEW_ZT_BBITS));
	     v++) {
		x_b_code.s.l = v;
		for (uint32_t r_v = 0; r_v < ((1 << PARALLELVIEW_ZT_BBITS)); r_v++) {
			x_b_code.s.r = r_v;
			if (not zt.branch_valid(x_b_code.int_v)) {
				continue;
			}

			const PVRow row = zt.get_branch_element(x_b_code.int_v, 0);
			return get_elided_text(
			    QString::fromStdString(_view.get_rushnraw_parent().at_string(row, _nraw_col)));
		}
	}

	return get_elided_text("None");
}

QString PVParallelView::PVScatterView::get_y_value_at(const qint64 value)
{
	PVParallelView::PVZoneTree const& zt = get_zone_tree();

	const PVCol nraw_col = lib_view().get_axes_combination().get_nraw_axis(get_zone_index() + 1);

	PVParallelView::PVBCode y_b_code;
	y_b_code.int_v = 0;
	// Look for all "upper" left value
	for (uint32_t v = (value >> (32 - PARALLELVIEW_ZT_BBITS)); v < ((1 << PARALLELVIEW_ZT_BBITS));
	     v++) {
		y_b_code.s.r = v;
		for (uint32_t r_v = 0; r_v < ((1 << PARALLELVIEW_ZT_BBITS)); r_v++) {
			y_b_code.s.l = r_v;
			if (not zt.branch_valid(y_b_code.int_v)) {
				continue;
			}

			const PVRow row = zt.get_branch_element(y_b_code.int_v, 0);
			return get_elided_text(
			    QString::fromStdString(_view.get_rushnraw_parent().at_string(row, nraw_col)));
		}
	}
	return get_elided_text("None");
}

////
// PVParallelView::PVScatterView::RenderedImage implementation
////

void PVParallelView::PVScatterView::RenderedImage::swap(QImage const& img,
                                                        QRectF const& viewport_rect,
                                                        QTransform const& mv2s)
{
	_mv2s = mv2s;
	_img = img.copy(viewport_rect.toAlignedRect()).mirrored(true, false);
}

void PVParallelView::PVScatterView::RenderedImage::draw(PVGraphicsView* view, QPainter* painter)
{
	const QTransform img_trans =
	    _mv2s * view->get_transform_from_scene() * view->get_transform_to_margined_viewport();
	// const QPainter::RenderHints hints = painter->renderHints();

	painter->save();
	painter->setTransform(img_trans, true);
	// painter->setRenderHints(hints | QPainter::SmoothPixmapTransform);
	painter->drawImage(QPointF(0.0, 0.0), _img);
	painter->restore();
}
