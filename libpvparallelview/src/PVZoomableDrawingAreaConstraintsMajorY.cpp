
#include <pvparallelview/PVZoomableDrawingAreaConstraintsMajorY.h>

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaConstraintsMajorY::zoom_x_available
 *****************************************************************************/

bool PVParallelView::PVZoomableDrawingAreaConstraintsMajorY::zoom_x_available() const
{
	return true;
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaConstraintsMajorY::zoom_y_available
 *****************************************************************************/

bool PVParallelView::PVZoomableDrawingAreaConstraintsMajorY::zoom_y_available() const
{
	return true;
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaConstraintsMajorY::set_zoom_value
 *****************************************************************************/

bool PVParallelView::PVZoomableDrawingAreaConstraintsMajorY::set_zoom_value(int axes, int value,
	      PVParallelView::PVAxisZoom &zx,
	      PVParallelView::PVAxisZoom &zy)
{
	if (axes & PVParallelView::PVZoomableDrawingAreaConstraints::X) {
		set_clamped_value(zx, value);
	}
	if (axes & PVParallelView::PVZoomableDrawingAreaConstraints::Y) {
		set_clamped_value(zy, value);
	}
	return true;
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaConstraintsMajorY::increment_zoom_value
 *****************************************************************************/

bool PVParallelView::PVZoomableDrawingAreaConstraintsMajorY::increment_zoom_value(int axes, int value,
	            PVParallelView::PVAxisZoom &zx,
	            PVParallelView::PVAxisZoom &zy)
{
	if (axes & PVParallelView::PVZoomableDrawingAreaConstraints::X) {
		set_clamped_value(zx, zx.get_value() + value);
	}
	if (axes & PVParallelView::PVZoomableDrawingAreaConstraints::Y) {
		set_clamped_value(zy, zy.get_value() + value);
	}
	return true;
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaConstraintsMajorY::adjust_pan
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingAreaConstraintsMajorY::adjust_pan(QScrollBar64 */*xsb*/,
	  QScrollBar64 */*ysb*/)
{}
