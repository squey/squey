/**
 * \file PVSelectionRectangleItem.h
 *
 * Copyright (C) Picviz Labs 2013
 */

#ifndef PVWIDGETS_PVSELECTIONRECTANGLEITEM_H
#define PVWIDGETS_PVSELECTIONRECTANGLEITEM_H

#include <QGraphicsItem>
#include <QCursor>
#include <QPen>

#include <vector>

namespace PVParallelView
{

class PVSelectionHandleItem;

/**
 * @class PVSelectionRectangleItem
 *
 * represent a selection rectangle as the Gimp provides.
 *
 * @note as handles can be resizable in one, two or zero dimension, the
 * view->scene transformation matrix (at least the scale factors) must be
 * passed to the selection rectangle.
 *
 * @note a QGraphicsScene using a PVSelectionRectangleItem can also not be shared
 * between many QGraphicsView.
 */
class PVSelectionRectangleItem : public QGraphicsObject
{
	Q_OBJECT

	friend PVSelectionHandleItem;

public:
	enum SelectionMode
	{
		RECTANGLE,
		HORIZONTAL,
		VERTICAL
	};

public:
	static constexpr qreal MOVE_STEP_PX = 1;
	static constexpr qreal GROW_STEP_RATIO = 1.2;

public:
	PVSelectionRectangleItem(const QRectF& rect = QRectF(),
	                         QGraphicsItem* parent = nullptr);

public:
	/**
	 * clear and hide the selection rectangle
	 */
	void clear();

	/**
	 * show and start a mouse interaction
	 */
	void begin(const QPointF& p);

	/**
	 * process a mouse interaction step
	 */
	void step(const QPointF& p);

	/**
	 * terminate a mouse interaction
	 */
	void end(const QPointF& p);

public:
	/**
	 * set the selection mode
	 *
	 * @param sel_mode the selection mode
	 */
	void set_selection_mode(int sel_mode);

	/**
	 * get the selection mode
	 *
	 * @return the selection mode
	 */
	SelectionMode selection_mode() const
	{
		return _sel_mode;
	}

public:
	/**
	 * set the pen color to use to draw the rectangle
	 *
	 * @param color the color to use
	 */
	void set_pen_color(const QColor& col);

	/**
	 * set the default mouse cursor to use
	 *
	 * @param cursor the default mouse cursor
	 */
	void set_default_cursor(QCursor cursor);

	/**
	 * get the default mouse cursor to use
	 *
	 * @return cursor the default mouse cursor
	 */
	QCursor get_default_cursor() const;

	/**
	 * set the pen color to use to draw the handles outline
	 *
	 * @param color the color to use
	 */
	void set_handles_pen_color(const QColor& col) const;

	/**
	 * set the pen color to use to fill the handles
	 *
	 * using a color with an alpha != 0
	 *
	 * @param color the color to use
	 */
	void set_handles_brush_color(const QColor& col) const;

	/**
	 * set the scale factors for handles
	 *
	 * @sa PVSelectionHandle::set_scale(const qreal xscale, const qreal yscale)
	 *
	 * @param xscale the scale factor along X
	 * @param yscale the scale factor along Y
	 */
	void set_handles_scale(const qreal xscale, const qreal yscale) const;

	/**
	 * get the handles x scale factor
	 *
	 * @return the x scale factor
	 */
	qreal get_handles_x_scale() const;

	/**
	 * get the handles y scale factor
	 *
	 * @return the y scale factor
	 */
	qreal get_handles_y_scale() const;

	/**
	 * set the rectangle's geometry
	 *
	 * @param rect the rectangle
	 */
	void set_rect(const QRectF& rect);

	/**
	 * get the selection rectangle's geometry
	 *
	 * @return the reactangle's geometry
	 */
	QRectF get_rect();

	/**
	 * get the selection rectangle's geometry (const version)
	 *
	 * @return the reactangle's geometry
	 */
	const QRectF get_rect() const;

	/**
	 * set the horizontal range for the rectangle
	 *
	 * @param min_value the lower bound value
	 * @param max_value the upper bound value
	 */
	void set_x_range(qreal min_value, qreal max_value);

	/**
	 * set the vertical range for the rectangle
	 *
	 * @param min_value the lower bound value
	 * @param max_value the upper bound value
	 */
	void set_y_range(qreal min_value, qreal max_value);

	/**
	 * clear the horizontal range for the rectangle
	 */
	void clear_x_range();

	/**
	 * clear the vertical range for the rectangle
	 */
	void clear_y_range();

	/**
	 * get the horizontal lower bound
	 *
	 * @return the horizontal lower bound
	 */
	qreal get_x_min() const
	{
		return _x_min_value;
	}

	/**
	 * get the horizontal upper bound
	 *
	 * @return the horizontal upper bound
	 */
	qreal get_x_max() const
	{
		return _x_max_value;
	}

	/**
	 * get the vertical lower bound
	 *
	 * @return the vertical lower bound
	 */
	qreal get_y_min() const
	{
		return _y_min_value;
	}

	/**
	 * get the vertical upper bound
	 *
	 * @return the vertical upper bound
	 */
	qreal get_y_max() const
	{
		return _y_max_value;
	}

public:
	/**
	 * get the selection rectangle's bounding box
	 *
	 * @reimpl QGraphicsItem::boundingRect()
	 *
	 * @return the selection rectangle's bounding box
	 */
	QRectF boundingRect() const override;

	/**
	 * paint the selection rectangle
	 *
	 * @reimpl QGraphicsItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
	 *
	 * @param painter the current painter
	 * @param option the current style
	 * @param widget the current QWidget
	 */
	void paint(QPainter* painter,
	           const QStyleOptionGraphicsItem* option,
	           QWidget* widget) override;

signals:
	void geometry_has_changed(const QRectF& old_rect, const QRectF& new_rect);

protected:
	/**
	 * make the handles been moved to the rectangle's scene.
	 *
	 * @param change the change's type
	 * @param value the change's value
	 */
	QVariant itemChange(GraphicsItemChange change, const QVariant& value) override;

protected:
	/**
	 * update all handles
	 */
	void update_handles() const;

	/**
	 * set the cursor used by all handles
	 *
	 * @param cursor the cursor to use
	 */
	void set_handles_cursor(const QCursor& cursor);

	/**
	 * reset the handles cursors to their default value
	 */
	void reset_handles_cursor();

	/**
	 * make all handles visible
	 */
	void show_all_handles() const;

	/**
	 * make all handles invisible
	 *
	 * a synonym for PVSelectionHandle::hide_all_handles_but(nullptr)
	 *
	 * @sa PVSelectionHandle::hide_all_handles_but(PVSelectionHandle* handle)
	 */
	void hide_all_handles() const;

	/**
	 * make all handles invisible except @a handle
	 *
	 * this method can be called with any value, if @a handle is not a valid
	 * handle, all handles will be hidden.
	 *
	 * @param handle the handle to keep visible
	 */
	void hide_all_handles_but(PVSelectionHandleItem* handle) const;

private:
	std::vector<PVSelectionHandleItem*> _handles;
	QCursor                             _default_cursor;
	QPen                                _pen;
	QRectF                              _rect;
	QPointF                             _ref;
	qreal                               _x_min_value;
	qreal                               _x_max_value;
	qreal                               _y_min_value;
	qreal                               _y_max_value;
	SelectionMode                       _sel_mode;
};

}

#endif // PVWIDGETS_PVSELECTIONRECTANGLEITEM_H
