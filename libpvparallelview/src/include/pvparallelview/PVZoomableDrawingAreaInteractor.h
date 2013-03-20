#ifndef PVPARALLELVIEW_PVZOOMABLEDRAWINGAREAINTERACTOR_H
#define PVPARALLELVIEW_PVZOOMABLEDRAWINGAREAINTERACTOR_H

#include <pvkernel/widgets/PVGraphicsViewInteractor.h>
#include <pvparallelview/PVZoomableDrawingArea.h>

namespace PVParallelView {

/**
 * @class PVZoomableDrawingAreaInteractor
 */

class PVZoomableDrawingAreaInteractor : public PVWidgets::PVGraphicsViewInteractor<PVZoomableDrawingArea>
{
public:
	PVZoomableDrawingAreaInteractor(PVWidgets::PVGraphicsView* parent = nullptr) :
		PVWidgets::PVGraphicsViewInteractor<PVZoomableDrawingArea>(parent)
	{}

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
	static inline bool set_zoom_value(PVZoomableDrawingArea *zda, int axes, int value)
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
	static inline bool increment_zoom_value(PVZoomableDrawingArea *zda, int axes, int value)
	{
		return zda->increment_zoom_value(axes, value);
	}

	static inline void zoom_has_changed(PVZoomableDrawingArea *zda)
	{
		emit zda->zoom_has_changed();
	}

	static inline void pan_has_changed(PVZoomableDrawingArea *zda)
	{
		emit zda->pan_has_changed();
	}
};

}

#endif // PVPARALLELVIEW_PVZOOMABLEDRAWINGAREAINTERACTOR_H
