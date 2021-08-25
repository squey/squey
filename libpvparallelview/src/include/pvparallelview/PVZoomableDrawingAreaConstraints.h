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

#ifndef PVPARALLELVIEW_PVZOOMABLEDRAWINGAREACONSTRAINTS_H
#define PVPARALLELVIEW_PVZOOMABLEDRAWINGAREACONSTRAINTS_H

#include <pvparallelview/PVAxisZoom.h>

class QScrollBar;

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
	typedef enum { X = 1, Y = 2 } AxisMask;

  public:
	/**
	 * Needed virtual DTOR to compile
	 */
	virtual ~PVZoomableDrawingAreaConstraints() = default;

  public:
	/**
	 * Returns if a zoom operation is available on axis X
	 *
	 * @return true if a zoom operation is available on axis X, false otherwise
	 */
	virtual bool zoom_x_available() const = 0;

	/**
	 * Returns if a zoom operation is available on axis Y
	 *
	 * @return true if a zoom operation is available on axis Y, false otherwise
	 */
	virtual bool zoom_y_available() const = 0;

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
	virtual bool set_zoom_value(int axes, int value, PVAxisZoom& zx, PVAxisZoom& zy) = 0;

	/**
	 * Change the zoom value given its parameters
	 *
	 * The @a value parameter is added to the current stored value.
	 *
	 * @param axes [in] an axis mask (see @ref AxisMask) to tell which axis will be affected
	 * @param value [in] the value to add to the zoom value
	 * @param zx [in] the PVAxisZoom of X axis
	 * @param zy [in] the PVAxisZoom of Y axis
	 * @return true if a change has occur, false otherwise
	 */
	virtual bool increment_zoom_value(int axes, int value, PVAxisZoom& zx, PVAxisZoom& zy) = 0;

	/**
	 * Make adjustment to the PVGraphicsView's scrollbars according to
	 * internal state.
	 *
	 * @param xsb the horizontal view's scrollbar
	 * @param ysb the vertical view's scrollbar
	 */
	virtual void adjust_pan(QScrollBar* xsb, QScrollBar* ysb) = 0;

  protected:
	/**
	 * Set current value of @a az to @a value
	 *
	 * Why this method? Simply because friendship can not be inherited in C++.
	 *
	 * @param az the PVAxisZoom to update
	 * @param value [in] the new value.
	 */
	static inline void set_value(PVParallelView::PVAxisZoom& az, int value) { az.set_value(value); }

	/**
	 * Set current value of @a az to @a value
	 *
	 * The value is clamped before it is set.
	 *
	 * Why this method? Simply because friendship can not be inherited in C++.
	 *
	 * @param az the PVAxisZoom to update
	 * @param value [in] the new value.
	 */
	static inline void set_clamped_value(PVParallelView::PVAxisZoom& az, int value)
	{
		az.set_clamped_value(value);
	}
};
} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVZOOMABLEDRAWINGAREACONSTRAINTS_H
