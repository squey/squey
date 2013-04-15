
#include <pvparallelview/PVZoomableDrawingAreaConstraintsHomothetic.h>

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaConstraintsHomothetic::zoom_x_available
 *****************************************************************************/

bool PVParallelView::PVZoomableDrawingAreaConstraintsHomothetic::zoom_x_available() const
{
	return true;
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaConstraintsHomothetic::zoom_y_available
 *****************************************************************************/

bool PVParallelView::PVZoomableDrawingAreaConstraintsHomothetic::zoom_y_available() const
{
	return true;
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaConstraintsHomothetic::set_zoom_value
 *****************************************************************************/

bool PVParallelView::PVZoomableDrawingAreaConstraintsHomothetic::set_zoom_value(int /*axes*/, int value,
                                                                                PVParallelView::PVAxisZoom &zx,
                                                                                PVParallelView::PVAxisZoom &zy)
{
	set_clamped_value(zx, value);
	set_clamped_value(zy, value);
	return true;
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaConstraintsHomothetic::increment_zoom_value
 *****************************************************************************/

bool PVParallelView::PVZoomableDrawingAreaConstraintsHomothetic::increment_zoom_value(int /*axes*/, int value,
                                                                                      PVParallelView::PVAxisZoom &zx,
                                                                                      PVParallelView::PVAxisZoom &zy)
{
	set_clamped_value(zx, zx.get_value() + value);
	set_clamped_value(zy, zy.get_value() + value);
	return true;
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaConstraintsHomothetic::adjust_pan
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingAreaConstraintsHomothetic::adjust_pan(QScrollBar64 */*xsb*/,
	                                                                    QScrollBar64 */*ysb*/)
{}
