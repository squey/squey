//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

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

bool PVParallelView::PVZoomableDrawingAreaConstraintsHomothetic::set_zoom_value(
    int /*axes*/, int value, PVParallelView::PVAxisZoom& zx, PVParallelView::PVAxisZoom& zy)
{
	set_clamped_value(zx, value);
	set_clamped_value(zy, value);
	return true;
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaConstraintsHomothetic::increment_zoom_value
 *****************************************************************************/

bool PVParallelView::PVZoomableDrawingAreaConstraintsHomothetic::increment_zoom_value(
    int /*axes*/, int value, PVParallelView::PVAxisZoom& zx, PVParallelView::PVAxisZoom& zy)
{
	set_clamped_value(zx, zx.get_value() + value);
	set_clamped_value(zy, zy.get_value() + value);
	return true;
}

/*****************************************************************************
 * PVParallelView::PVZoomableDrawingAreaConstraintsHomothetic::adjust_pan
 *****************************************************************************/

void PVParallelView::PVZoomableDrawingAreaConstraintsHomothetic::adjust_pan(QScrollBar* /*xsb*/,
                                                                            QScrollBar* /*ysb*/)
{
}
