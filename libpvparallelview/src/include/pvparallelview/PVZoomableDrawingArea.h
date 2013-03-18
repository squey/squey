
#ifndef PVPARALLELVIEW_PVZOOMABLEDRAWINGAREA_H
#define PVPARALLELVIEW_PVZOOMABLEDRAWINGAREA_H

#include <assert.h>

#include <pvkernel/widgets/PVGraphicsView.h>

class QWidget;
class QGraphicsScene;
class QMouseEvent;
class QWheelEvent;

namespace PVParallelView
{

/**
 * @class PVZoomableDrawingArea
 *
 * A canvas to display a QGraphicsScene with few conveniences:
 * - zoom and pan are already implemented, you just have to define
 *   the scene's bounding box, the range of wheel value and the
 *   convertion method between wheel and the zoom effect;
 * - zoom and pan can be enabled/disabled along each axes.
 *
 * @see zoomable_drazwing_area_test.cpp for a minimalist zoomed
 * parallel view widget.
 *
 * @todo update this comment
 */

namespace __impl
{

typedef enum {
	XIsNeeded = 1,
	YIsNeeded = 2,
	Bound     = 4
} AxesPolicyValue;

}

class PVZoomableDrawingArea : public PVWidgets::PVGraphicsView
{
	Q_OBJECT

public:
	typedef enum {
		AlongNone    = 0,
		AlongX       = __impl::XIsNeeded,
		AlongY       = __impl::YIsNeeded,
		AlongBoth    = __impl::XIsNeeded | __impl::YIsNeeded,
		// Bound     = __impl::Bound
		BoundMajorX  = __impl::Bound | __impl::XIsNeeded,
		BoundMajorY  = __impl::Bound | __impl::YIsNeeded

	} AxesPolicy;

public:
	/**
	 * The Constructor
	 */
	PVZoomableDrawingArea(QWidget *parent = nullptr);

	/**
	 * The Destructor :-)
	 */
	~PVZoomableDrawingArea();

	/**
	 * Set how the zoom controls affect the zoom transformation.
	 *
	 * @param policy @see AxesPolicy
	 */
	void set_zoom_policy(const AxesPolicy policy)
	{
		_zoom_policy = policy;
	}

	/**
	 * Return the current zoom policy
	 *
	 * @return the zoom policy
	 */
	AxesPolicy get_zoom_policy() const
	{
		return (AxesPolicy)_zoom_policy;
	}

public:
	/**
	 * apply adjustments on zoom values.
	 *
	 * This method is call by ::update_zoom() before any transformation's component
	 * recomputation. Reimplement this method if you plan to add constraints on zoom
	 * values. In that case, force last parameter of ::set_zoom_value(), ::set_x_zoom_value()
	 * or ::set_y_zoom_value() to false to avoid looping update.
	 *
	 * By default, this method does nothing
	 */
	virtual void adjust_zoom_values();

	/**
	 * Set the current zoom value for all dimension
	 *
	 * It is clamped to respctive dimension's range.
	 *
	 * @param value the new value
	 * @param propagate_update to request or not an update of the view when a change occurs
	 */
	void set_zoom_value(const qint64 value, const bool propagate_update = true);


	/**
	 * Set the current zoom value for X dimension
	 *
	 * It is clamped to its range.
	 *
	 * @param value the new value
	 * @param propagate_update to request or not an update of the view when a change occurs
	 */
	void set_x_zoom_value(const qint64 value, const bool propagate_update = true);

	/**
	 * Set the current zoom value for Y dimension
	 *
	 * It is clamped to its range.
	 *
	 * @param value the new value
	 * @param propagate_update to request or not an update of the view when a change occurs
	 */
	void set_y_zoom_value(const qint64 value, const bool propagate_update = true);

	/**
	 * Return the current zoom value
	 *
	 * @return the current zoom value
	 */
	qint64 get_zoom_value() const
	{
		/**
		 * as _zoom_policy is a bitfield, there are 2 cases:
		 * - need return _x_zoom_value when it is AlongX, AlongBoth and BoundMajorX:
		 *   - AlongX because we zoom only on X
		 *   - AlongBoth because we zoom on both axes which remaind equal over time
		 *   - BoundMajorX because X is the main axis
		 * - need return _x_zoom_value when it is AlongNone, AlongY and BoundMajorX
		 *   - AlongNone because value is normally 0
		 *   - AlongY because we zoom only on Y
		 *   - BoundMajorY because Y is the main axis
		 */
		if (_zoom_policy & AlongX) {
			return _x_zoom_value;
		} else {
			return _y_zoom_value;
		}
	}

	/**
	 * Return the current zoom value along x
	 *
	 * @return the current zoom value
	 */
	qint64 get_x_zoom_value() const
	{
		return _x_zoom_value;
	}

	/**
	 * Return the current zoom value along y
	 *
	 * @return the current zoom value
	 */
	qint64 get_y_zoom_value() const
	{
		return _y_zoom_value;
	}

	/**
	 * Return the current zoom value relatively to the lowest zoom value
	 *
	 * @return the current relative zoom value
	 */
	qint64 get_relative_zoom_value() const
	{
		/**
		 * same remark than ::get_zoom_value()
		 */
		if (_zoom_policy & AlongX) {
			return _x_zoom_value - _x_zoom_min;
		} else {
			return _x_zoom_value - _y_zoom_min;
		}
	}

	/**
	 * Return the current x zoom value relatively to the lowest x zoom value
	 *
	 * @return the current relative x zoom value
	 */
	qint64 get_relative_x_zoom_value() const
	{
		return _x_zoom_value - _x_zoom_min;
	}

	/**
	 * Return the current y zoom value relatively to the lowest y zoom value
	 *
	 * @return the current relative y zoom value
	 */
	qint64 get_relative_y_zoom_value() const
	{
		return _y_zoom_value - _y_zoom_min;
	}

	/**
	 * Set the range of inclusive valid values for zoom along both axes.
	 *
	 * @param z_min the inclusive lowest value
	 * @param z_max the inclusive highest value
	 * @param propagate_update to request or not an update of the view when a change occurs
	 */
	void set_zoom_range(const qint64 z_min, const qint64 z_max, const bool propagate_update = true);

	/**
	 * Set the range of inclusive valid values for zoom along X.
	 *
	 * @param z_min the inclusive lowest value
	 * @param z_max the inclusive highest value
	 * @param propagate_update to request or not an update of the view when a change occurs
	 */
	void set_x_zoom_range(const qint64 z_min, const qint64 z_max, const bool propagate_update = true);

	/**
	 * Set the range of inclusive valid values for zoom along Y.
	 *
	 * @param z_min the inclusive lowest value
	 * @param z_max the inclusive highest value
	 * @param propagate_update to request or not an update of the view when a change occurs
	 */
	void set_y_zoom_range(const qint64 z_min, const qint64 z_max, const bool propagate_update = true);

	/**
	 * Return the lowest zoom's value.
	 *
	 * depend on zoom policy
	 *
	 * @return the lowest zoom's value
	 */
	qint64 get_zoom_min() const
	{
		/**
		 * same remark than ::get_zoom_value()
		 */
		if (_zoom_policy & AlongX) {
			return _x_zoom_min;
		} else {
			return _y_zoom_min;
		}
	}

	/**
	 * Return the highest zoom's value.
	 *
	 * depend on zoom policy
	 *
	 * @return the highest zoom's value
	 */
	qint64 get_zoom_max() const
	{
		/**
		 * same remark than ::get_zoom_value()
		 */
		if (_zoom_policy & AlongX) {
			return _x_zoom_max;
		} else {
			return _y_zoom_max;
		}
	}

	/**
	 * Return the lowest zoom's value along X
	 *
	 * @return the lowest zoom's value
	 */
	qint64 get_x_zoom_min() const
	{
		return _x_zoom_min;
	}

	/**
	 * Return the highest zoom's value along Y
	 *
	 * @return the highest zoom's value
	 */
	qint64 get_x_zoom_max() const
	{
		return _x_zoom_max;
	}

	/**
	 * Return the lowest zoom's value along Y
	 *
	 * @return the lowest zoom's value
	 */
	qint64 get_y_zoom_min() const
	{
		return _y_zoom_min;
	}

	/**
	 * Return the highest zoom's value along Y
	 *
	 * @return the highest zoom's value
	 */
	qint64 get_y_zoom_max() const
	{
		return _y_zoom_max;
	}

public:
	/**
	 * Set how the pan controls affect the pan transformation.
	 *
	 * @param policy @see AxesPolicy
	 */
	void set_pan_policy(const AxesPolicy policy)
	{
		switch (policy) {
		case AlongNone:
		case AlongX:
		case AlongY:
		case AlongBoth:
			_pan_policy = policy;
			break;
		default:
			assert(false);
		}
	}

	/**
	 * Return the current pan policy
	 *
	 * @return the pan policy
	 */
	AxesPolicy get_pan_policy() const
	{
		return (AxesPolicy)_pan_policy;
	}

protected:
	/**
	 * convert zoom value into scale factor
	 *
	 * @param wheel_value a valid zoom
	 */
	virtual qreal zoom_to_scale(const int zoom_value) const;

	/**
	 * convert x zoom value into scale factor
	 *
	 * default call to ::zoom_to_scale
	 *
	 * @param wheel_value a valid zoom
	 */
	virtual qreal x_zoom_to_scale(const int zoom_value) const;

	/**
	 * convert y zoom value into scale factor
	 *
	 * default call to ::zoom_to_scale
	 *
	 * @param wheel_value a valid zoom
	 */
	virtual qreal y_zoom_to_scale(const int zoom_value) const;

	/**
	 * convert zoom value into scale factor
	 */
	virtual int scale_to_zoom(const qreal scale_value) const;

	/**
	 * convert y zoom value into scale factor
	 *
	 * default call to ::scale_to_zoom
	 */
	virtual int y_scale_to_zoom(const qreal scale_value) const;

	/**
	 * convert x zoom value into scale factor
	 *
	 * default call to ::scale_to_zoom
	 */
	virtual int x_scale_to_zoom(const qreal scale_value) const;

	/**
	 * Returns a transformation given scale factors along x and y
	 *
	 * This method has to be overridden if you wish to define
	 * your own transformation when rendering the scene.
	 *
	 * The default behaviour is: zoom controls scale along axes
	 * which have zoom policy on and forces scene to fit in the
	 * viewport for those whose zoom policy is off.
	 *
	 * @param x_scale_value the scale value along x
	 * @param y_scale_value the scale value along y
	 *
	 * @return the transformation for scene to viewport rendering
	 */
	virtual QTransform scale_to_transform(const qreal x_scale_value,
	                                      const qreal y_scale_value) const;

protected:
	virtual void mousePressEvent(QMouseEvent *event);
	virtual void mouseReleaseEvent(QMouseEvent *event);
	virtual void mouseMoveEvent(QMouseEvent *event);
	virtual void resizeEvent(QResizeEvent *event);
	virtual void wheelEvent(QWheelEvent *event);

signals:
	/**
	 * This signal is emitted each time a zoom change has been done.
	 */
	void zoom_has_changed();

	/**
	 * This signal is emitted each time a pan change has been done.
	 */
	void pan_has_changed();

protected:
	/**
	 * Simply process from "zoom to scale" to "update widget".
	 */
	void update_zoom();

private:
	/**
	 * This method is volontarily private, so use the scene provided
	 * by ::get_scene()
	 */
	void set_scene(QGraphicsScene *scene)
	{
		PVWidgets::PVGraphicsView::set_scene(scene);
	}

	/**
	 * as tranformation updates may changed scrollbars values, those may
	 * have to be readjusted.
	 */
	void adjust_pan();

private:
	int        _zoom_policy;

	qint64     _x_zoom_min;
	qint64     _x_zoom_max;
	qint64     _x_zoom_value;

	qint64     _y_zoom_min;
	qint64     _y_zoom_max;
	qint64     _y_zoom_value;

	int        _pan_policy;
	QPoint     _pan_reference;
};

}

#endif // PVPARALLELVIEW_PVZOOMABLEDRAWINGAREA_H
