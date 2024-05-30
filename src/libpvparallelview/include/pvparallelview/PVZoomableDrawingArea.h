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

#ifndef PVPARALLELVIEW_PVZOOMABLEDRAWINGAREA_H
#define PVPARALLELVIEW_PVZOOMABLEDRAWINGAREA_H

#include <assert.h>

#include <pvkernel/widgets/PVGraphicsView.h>

#include <pvparallelview/PVAxisZoom.h>

/* must include it because MOC does shit
 * replace it with a forward declaration when signals will be
 * replaced with virtual methods
 */
#include <pvparallelview/PVZoomableDrawingAreaConstraints.h>

class QWidget;
class QGraphicsScene;
class QMouseEvent;
class QWheelEvent;

namespace PVParallelView
{

class PVZoomableDrawingAreaInteractor;

/**
 * @class PVZoomableDrawingArea
 *
 * A generic canvas to create views with few conveniences:
 * - customizable axis zoom value constraints using PVAxisZoom
 * - customizable mapping between zoom value and scale value using PVZoomConverter
 * - customizable zoom and pan controls using PVZoomableDrawingAreaInteractor
 * - customizable zoom and pan behaviours using PVZoomableDrawingAreaConstraints
 *
 * @attention the lone drawback of this system is that you have to make sure that
 * the PVZoomableDrawingAreaInteractor and the PVZoomableDrawingAreaConstraints
 * you use are compatible.
 */

class PVZoomableDrawingArea : public PVWidgets::PVGraphicsView
{
	Q_OBJECT

	friend class PVZoomableDrawingAreaInteractor;

  public:
	/**
	 * The Constructor
	 */
	explicit PVZoomableDrawingArea(QWidget* parent = nullptr);

	/**
	 * The Destructor :-)
	 */
	~PVZoomableDrawingArea() override;

  protected:
	bool set_zoom_value(int axes, int value)
	{
		assert(_constraints != nullptr);
		return _constraints->set_zoom_value(axes, value, _x_axis_zoom, _y_axis_zoom);
	}

	bool increment_zoom_value(int axes, int value)
	{
		assert(_constraints != nullptr);
		return _constraints->increment_zoom_value(axes, value, _x_axis_zoom, _y_axis_zoom);
	}

	PVAxisZoom& get_x_axis_zoom() { return _x_axis_zoom; }

	const PVAxisZoom& get_x_axis_zoom() const { return _x_axis_zoom; }

	PVAxisZoom& get_y_axis_zoom() { return _y_axis_zoom; }

	const PVAxisZoom& get_y_axis_zoom() const { return _y_axis_zoom; }

  protected:
	void set_constraints(PVZoomableDrawingAreaConstraints* constraints)
	{
		_constraints = constraints;
	}

	PVZoomableDrawingAreaConstraints* get_constraints()
	{
		assert(_constraints != nullptr);
		return _constraints;
	}

	const PVZoomableDrawingAreaConstraints* get_constraints() const
	{
		assert(_constraints != nullptr);
		return _constraints;
	}

	/**
	 * proxy function to retrieve scale factor along X
	 *
	 * @param value [in] the zoom value to convert
	 *
	 * @return the corresponding scale value
	 */
	qreal x_zoom_to_scale(const int value) const;

	/**
	 * proxy function to retrieve scale factor along Y
	 *
	 * @param value [in] the zoom value to convert
	 *
	 * @return the corresponding scale value
	 */
	qreal y_zoom_to_scale(const int value) const;

	/**
	 * proxy function to retrieve zoom factor along X
	 *
	 * @param value [in] the scale value to convert
	 *
	 * @return the corresponding zoom value
	 */
	int x_scale_to_zoom(const qreal value) const;

	/**
	 * proxy function to retrieve zoom factor along Y
	 *
	 * @param value [in] the scale value to convert
	 *
	 * @return the corresponding zoom value
	 */
	int y_scale_to_zoom(const qreal value) const;

  public:
	/**
	 * Simply process from "zoom to scale" to "update widget".
	 */
	void reconfigure_view();

	void set_x_axis_inverted(bool inverted);
	void set_y_axis_inverted(bool inverted);

	bool x_axis_inverted() const { return get_x_axis_zoom().inverted(); }
	bool y_axis_inverted() const { return get_y_axis_zoom().inverted(); }

  Q_SIGNALS:
	/**
	 * This signal is emitted each time a zoom change has been done.
	 *
	 * \param axes Axes for which the zoom value has changed
	 */
	void zoom_has_changed(int axes);

	/**
	 * This signal is emitted each time a pan change has been done.
	 */
	void pan_has_changed();

  private:
	/**
	 * This method is volontarily private, so use the scene provided
	 * by ::get_scene()
	 */
	void set_scene(QGraphicsScene* scene) { PVWidgets::PVGraphicsView::set_scene(scene); }

  private:
	PVAxisZoom _x_axis_zoom;
	PVAxisZoom _y_axis_zoom;

	PVZoomableDrawingAreaConstraints* _constraints;
};
} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVZOOMABLEDRAWINGAREA_H
