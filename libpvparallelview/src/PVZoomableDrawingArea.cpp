
#include <pvkernel/core/PVAlgorithms.h>

#include <pvparallelview/PVZoomableDrawingArea.h>

#include <QGraphicsScene>
#include <QScrollBar64>
#include <QMouseEvent>
#include <QWheelEvent>

#include <iostream>

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

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingArea::reconfigure_view
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingArea::reconfigure_view()
{
	QTransform transfo;

	transfo.scale(x_zoom_to_scale(get_x_axis_zoom().get_clamped_value()),
	              y_zoom_to_scale(get_y_axis_zoom().get_clamped_value()));
	set_transform(transfo);

	get_constraints()->adjust_pan(get_horizontal_scrollbar(),
	                              get_vertical_scrollbar());
}
