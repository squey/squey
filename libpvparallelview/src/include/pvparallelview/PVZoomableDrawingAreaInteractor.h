/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
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
	PVZoomableDrawingAreaInteractor(PVWidgets::PVGraphicsView* parent = nullptr)
	    : PVWidgets::PVGraphicsViewInteractor<PVZoomableDrawingArea>(parent)
	{
	}

  protected:
	virtual bool resizeEvent(PVParallelView::PVZoomableDrawingArea* zda,
	                         QResizeEvent* /*event*/) override
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
}

#endif // PVPARALLELVIEW_PVZOOMABLEDRAWINGAREAINTERACTOR_H
