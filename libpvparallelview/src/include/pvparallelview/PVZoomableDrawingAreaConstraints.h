
#ifndef PVPARALLELVIEW_PVZOOMABLEDRAWINGAREACONSTRAINTS_H
#define PVPARALLELVIEW_PVZOOMABLEDRAWINGAREACONSTRAINTS_H

class QScrollBar64;

namespace PVParallelView
{

class PVAxisZoom;

/**
 * @class PVZoomableDrawingAreaConstraints
 *
 * This class represents zoom values interact with each other.
 *
 * @see notes about PVZoomableDrawingArea
 */

class PVZoomableDrawingAreaConstraints
{
public:
	/**
	 * an enum to use as a bitfield with set_zoom_value() and increment_zoom_value()
	 */
	typedef enum {
		X = 1,
		Y = 2
	} AxisMask;

public:
	/**
	 * Returns if a zoom operation is available on axis X
	 *
	 * @return true if a zoom operation is available on axis X, false otherwise
	 */
	bool zoom_x_available() const = 0;

	/**
	 * Returns if a zoom operation is available on axis Y
	 *
	 * @return true if a zoom operation is available on axis Y, false otherwise
	 */
	bool zoom_y_available() const = 0;

	/**
	 * Change the zoom value given its parameters
	 *
	 * The @a value parameter replaces the current stored value.
	 *
	 * @param axes [in] an axis mask (see @ref AxisMask) to tell which axis will be affected
	 * @param value [in] the new zoom value
	 * @param zx [in] the PVAxisZoom of X axis
	 * @param zy [in] the PVAxisZoom of Y axis
	 * @return true if a change has occur, false otherwise
	 */
	bool set_zoom_value(int axes, int value, PVAxisZoom &zx, PVAxisZoom &zy) = 0;

	/**
	 * Change the zoom value given its parameters
	 *
	 * The @a value parameter is added to the current stored value.
	 *
	 * @param axes [in] an axis mask (see @ref AxisMask) to tell which axis will be affected
	 * @param value [in] the value to add to zoom value
	 * @param zx [in] the PVAxisZoom of X axis
	 * @param zy [in] the PVAxisZoom of Y axis
	 * @return true if a change has occur, false otherwise
	 */
	bool increment_zoom_value(int axes, int value, PVAxisZoom &zx, PVAxisZoom &zy) = 0;

	/**
	 * Make adjustment to the PVGraphicsView's scrollbars according to
	 * internal state.
	 *
	 * @param xsb the horizontal view's scrollbar
	 * @param ysb the vertical view's scrollbar
	 */
	void adjust_pan(QScrollBar64 *xsb, QScrollBar64 *ysb) = 0;
};

}

#endif // PVPARALLELVIEW_PVZOOMABLEDRAWINGAREACONSTRAINTS_H
