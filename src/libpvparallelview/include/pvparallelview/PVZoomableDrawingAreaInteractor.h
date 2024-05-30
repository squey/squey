/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVPARALLELVIEW_PVZOOMABLEDRAWINGAREAINTERACTOR_H
#define PVPARALLELVIEW_PVZOOMABLEDRAWINGAREAINTERACTOR_H

#include <pvkernel/widgets/PVGraphicsViewInteractor.h>
#include <pvparallelview/PVZoomableDrawingArea.h>

namespace PVParallelView
{

/**
 * @class PVZoomableDrawingAreaInteractor
 */

class PVZoomableDrawingAreaInteractor
    : public PVWidgets::PVGraphicsViewInteractor<PVZoomableDrawingArea>
{
  public:
	explicit PVZoomableDrawingAreaInteractor(PVWidgets::PVGraphicsView* parent = nullptr)
	    : PVWidgets::PVGraphicsViewInteractor<PVZoomableDrawingArea>(parent)
	{
	}

  protected:
	bool resizeEvent(PVParallelView::PVZoomableDrawingArea* zda, QResizeEvent* /*event*/) override
	{
		zda->reconfigure_view();
		pan_has_changed(zda);
		return true;
	}

  protected:
	/**
	 * Change the PVZoomableDrawingArea's zoom value given its parameters
	 *
	 * Why this method? Simply because friendship can not be inherited in C++.
	 *
	 * @param zda the PVZoomDrawingArea to update
	 * @param axes [in] an axis mask (see @ref AxisMask) to tell which axis will be affected
	 * @param value [in] the new value.
	 */
	static inline bool set_zoom_value(PVZoomableDrawingArea* zda, int axes, int value)
	{
		return zda->set_zoom_value(axes, value);
	}

	/**
	 * Change the PVZoomableDrawingArea's zoom value given its parameters
	 *
	 * Why this method? Simply because friendship can not be inherited in C++.
	 *
	 * @param zda the PVZoomDrawingArea to update
	 * @param axes [in] an axis mask (see @ref AxisMask) to tell which axis will be affected
	 * @param value [in] the value to add to the zoom value
	 */
	static inline bool increment_zoom_value(PVZoomableDrawingArea* zda, int axes, int value)
	{
		return zda->increment_zoom_value(axes, value);
	}

	/**
	 * Q_EMIT a PVZoomableDrawingArea::zoom_has_changed signal
	 *
	 * Why this method? Simply because friendship can not be inherited in C++.
	 *
	 * @param zda  the corresponding PVZoomDrawingArea
	 * @param axes axes for which the zoom value has changed
	 */
	static inline void zoom_has_changed(PVZoomableDrawingArea* zda, int axes)
	{
		Q_EMIT zda->zoom_has_changed(axes);
	}

	/**
	 * Q_EMIT a PVZoomableDrawingArea::pan_has_changed signal
	 *
	 * Why this method? Simply because friendship can not be inherited in C++.
	 *
	 * @param zda the corresponding PVZoomDrawingArea
	 */
	static inline void pan_has_changed(PVZoomableDrawingArea* zda)
	{
		Q_EMIT zda->pan_has_changed();
	}
};
} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVZOOMABLEDRAWINGAREAINTERACTOR_H
