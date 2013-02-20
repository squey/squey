
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
 */

class PVZoomableDrawingArea : public PVWidgets::PVGraphicsView
{
public:
	typedef enum {
		AlongNone = 0,
		AlongX    = 1,
		AlongY    = 2,
		AlongBoth = 3
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

	/**
	 * Set the current zoom value
	 *
	 * It is clamped to its range.
	 *
	 * @param value the new value
	 */
	void set_zoom_value(const qint64 value);

	/**
	 * Return the current zoom value
	 *
	 * @return the current zoom value
	 */
	qint64 get_zoom_value() const
	{
		return _zoom_value;
	}

	/**
	 * Return the current zoom value relatively to the lowest zoom value
	 *
	 * @return the current relative zoom value
	 */
	qint64 get_relative_zoom_value() const
	{
		return _zoom_value - _zoom_min;
	}

	/**
	 * Set the range of inclusive valid values for zoom.
	 *
	 * @param z_min the inclusive lowest value
	 * @param z_max the inclusive highest value
	 */
	void set_zoom_range(const qint64 z_min, const qint64 z_max);

	/**
	 * Return the lowest zoom's value
	 *
	 * @return the lowest zoom's value
	 */
	qint64 get_zoom_min() const
	{
		return _zoom_min;
	}

	/**
	 * Return the highest zoom's value
	 *
	 * @return the highest zoom's value
	 */
	qint64 get_zoom_max() const
	{
		return _zoom_max;
	}

	/**
	 * Set how the pan controls affect the pan transformation.
	 *
	 * @param policy @see AxesPolicy
	 */
	void set_pan_policy(const AxesPolicy policy)
	{
		_pan_policy = policy;
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
	 * convert zoom value into scale factor
	 */
	virtual int scale_to_zoom(const qreal scale_value) const;

	virtual void mousePressEvent(QMouseEvent *event);
	virtual void mouseReleaseEvent(QMouseEvent *event);
	virtual void mouseMoveEvent(QMouseEvent *event);
	virtual void resizeEvent(QResizeEvent *event);
	virtual void wheelEvent(QWheelEvent *event);

private:
	void set_scene(QGraphicsScene *scene)
	{
		PVWidgets::PVGraphicsView::set_scene(scene);
	}

	void update_zoom();
	void update_pan();

private:
	int        _zoom_policy;
	qint64     _zoom_min;
	qint64     _zoom_max;
	qint64     _zoom_value;

	int        _pan_policy;
	QPoint     _pan_reference;
};

}

#endif // PVPARALLELVIEW_PVZOOMABLEDRAWINGAREA_H
