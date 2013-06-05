
#include <pvkernel/core/PVAlgorithms.h>

#include <pvparallelview/PVZoomableDrawingArea.h>

#include <QGraphicsScene>
#include <QScrollBar64>
#include <QMouseEvent>
#include <QWheelEvent>

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingArea::PVZoomableDrawingArea
 *****************************************************************************/

PVParallelView::PVZoomableDrawingArea::PVZoomableDrawingArea(QWidget *parent) :
	PVWidgets::PVGraphicsView(parent)
{
	set_transformation_anchor(AnchorUnderMouse);
	set_resize_anchor(AnchorViewCenter);

	QGraphicsScene *scene = new QGraphicsScene();
	set_scene(scene);
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingArea::~PVZoomableDrawingArea
 *****************************************************************************/

PVParallelView::PVZoomableDrawingArea::~PVZoomableDrawingArea()
{
	get_scene()->deleteLater();
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingArea::x_zoom_to_scale
 *****************************************************************************/

qreal PVParallelView::PVZoomableDrawingArea::x_zoom_to_scale(const int value) const
{
	return get_x_axis_zoom().get_zoom_converter()->zoom_to_scale(value);
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingArea::y_zoom_to_scale
 *****************************************************************************/

qreal PVParallelView::PVZoomableDrawingArea::y_zoom_to_scale(const int value) const
{
	return get_y_axis_zoom().get_zoom_converter()->zoom_to_scale(value);
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingArea::x_scale_to_zoom
 *****************************************************************************/

int PVParallelView::PVZoomableDrawingArea::x_scale_to_zoom(const qreal value) const
{
	return get_x_axis_zoom().get_zoom_converter()->scale_to_zoom(value);
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingArea::y_scale_to_zoom
 *****************************************************************************/

int PVParallelView::PVZoomableDrawingArea::y_scale_to_zoom(const qreal value) const
{
	return get_y_axis_zoom().get_zoom_converter()->scale_to_zoom(value);
}

void PVParallelView::PVZoomableDrawingArea::set_x_axis_inverted(bool inverted)
{
	get_x_axis_zoom().set_inverted(inverted);
	reconfigure_view();
}

void PVParallelView::PVZoomableDrawingArea::set_y_axis_inverted(bool inverted)
{
	get_y_axis_zoom().set_inverted(inverted);
	reconfigure_view();
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingArea::reconfigure_view
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingArea::reconfigure_view()
{
	if (!get_x_axis_zoom().valid() || !get_x_axis_zoom().valid()) {
		return;
	}

	QTransform transfo;

	transfo.scale(x_zoom_to_scale(get_x_axis_zoom().get_clamped_value()),
	              y_zoom_to_scale(get_y_axis_zoom().get_clamped_value()));

	const QRectF scene_rect = get_scene()->sceneRect();

	QTransform transfo_inv_x;
	QTransform transfo_inv_y;
	if (get_y_axis_zoom().inverted()) {
		transfo_inv_y.translate(0.0, scene_rect.height());
		transfo_inv_y.scale(1.0, -1.0);
	}
	if (get_x_axis_zoom().inverted()) {
		transfo_inv_x.translate(scene_rect.width(), 0.0);
		transfo_inv_x.scale(-1.0, 1.0);
	}

	set_transform(transfo_inv_x * transfo_inv_y * transfo);

	get_constraints()->adjust_pan(get_horizontal_scrollbar(),
	                              get_vertical_scrollbar());
}
