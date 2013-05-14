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
	_view(*pvview_sp),
	_zt(zm.get_zone_tree<PVParallelView::PVZoneTree>(axis_index)),
	_zzt(zm.get_zone_tree<PVParallelView::PVZoomedZoneTree>(axis_index)),
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
 * PVParallelView::PVScatterView::keyPressEvent
 *****************************************************************************/
void PVParallelView::PVScatterView::keyPressEvent(QKeyEvent* event)
	{
#ifdef PICVIZ_DEVELOPER_MODE
		if ((event->key() == Qt::Key_B) && (event->modifiers() & Qt::ControlModifier)) {
			PVScatterView::toggle_show_quadtrees();
		}
#endif
		PVZoomableDrawingAreaWithAxes::keyPressEvent(event);
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
void PVParallelView::PVScatterView::draw_points(QPainter* painter, const QRectF& /*margined_rect*/)
{
	Picviz::PVSelection const& sel = _view.get_real_output_selection();

	PVZoomedZoneTree::context_t ctxt;

	static PVCore::PVHSVColor* output_image = new PVCore::PVHSVColor[image_width*image_height];
	memset(output_image, HSV_COLOR_BLACK, image_width*image_height);

	int rel_zoom = get_y_axis_zoom().get_clamped_relative_value();

	QRectF view_rect = get_scene_rect().intersected(map_to_scene(get_margined_viewport_rect()));

	uint64_t y1_min = view_rect.x();
	uint64_t y1_max = view_rect.x()+view_rect.width();
	uint64_t y2_min = view_rect.y();
	uint64_t y2_max = view_rect.y()+view_rect.height();

	double alpha = 0.5 * _zoom_converter->zoom_to_scale_decimal(rel_zoom);

	const PVCore::PVHSVColor* colors = _view.output_layer.get_lines_properties().get_buffer();

	_zzt.browse_bci_by_y1_y2(
		ctxt,
		y1_min,
		y1_max,
		y2_min,
		y2_max,
		(rel_zoom/zoom_steps) + 1,
		alpha,
		colors,
		output_image
	);

	BENCH_START(image_convertion);
	static QImage* img = new QImage(image_width, image_height, QImage::Format_RGB32);
	QRgb* image_rgb = (QRgb*) &img->scanLine(0)[0];
#pragma omp parallel for schedule(static, 16)
	for (uint32_t i = 0; i < image_width*image_height; i++) {
		output_image[i].to_rgb((uint8_t*) &image_rgb[i]);
	}
	BENCH_END(image_convertion, "image_convertion", image_width*image_height, sizeof(PVCore::PVHSVColor), image_width*image_height, sizeof(QRgb));

	painter->drawImage(QPointF(0,0), *img);

	painter->setPen(Qt::white);
	painter->setOpacity(1.0);

	/*for (uint32_t i = 0; i < bci_count; ++i) {
		bcicode_t code_bci = bcicodes[i];
		QColor c =_view.get_color_in_output_layer(code_bci.s.idx).toQColor();
		painter->setPen(c);
		painter->setOpacity(sel.get_line_fast(code_bci.s.idx) ? 1.0 : 0.25);
		painter->drawPoint(code_bci.s.l, code_bci.s.r);
	}*/

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

				QPointF view_point = map_margined_from_scene(QPointF(x_scene, y_scene));
				QPointF view_point_rect = map_margined_from_scene(QPointF(x_rect_scene, y_rect_scene));

				painter->setPen(_view.get_color_in_output_layer(row).toQColor());
				painter->setOpacity(sel.get_line_fast(row) ? 1.0 : 0.25);
				painter->drawRect(QRectF(view_point, view_point_rect));
			}
		}
	}
}
