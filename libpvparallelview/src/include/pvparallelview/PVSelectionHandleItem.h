/**
 * \file PVSelectionHandleItem.h
 *
 * Copyright (C) Picviz Labs 2013
 */

#ifndef PVWIDGETS_PVSELECTIONHANDLEITEM_H
#define PVWIDGETS_PVSELECTIONHANDLEITEM_H

#include <QGraphicsItem>
#include <QCursor>
#include <QPen>
#include <QBrush>
#include <QColor>

namespace PVParallelView
{

class PVSelectionRectangleItem;

/**
 * @class PVSelectionHandleItem
 *
 * represent a selection rectangle's handle.
 *
 * @note  As QGraphicsItem::setVisible() can not be overridden as in QWidget, all the
 * logic to visually hide/show the handle has been added.
 *
 * @note As we rewrite hide/show logic, forcing always hidden" mode is trickier: hover
 * events, used mouse cursor and more complex visibility setting.
 */
class PVSelectionHandleItem : public QGraphicsItem
{
	friend PVSelectionRectangleItem;

public:
	static constexpr int handle_size = 11;

	typedef enum {
		N = 1,
		S = 2,
		E = 4,
		W = 8,
		NE = N | E,
		SE = S | E,
		SW = S | W,
		NW = N | W,
		CENTER = N | S | E | W
	} handle_type;

public:
	/**
	 * constructor
	 *
	 * @param type the handle's type
	 * @param sel_rect the associated selection rectangle
	 */
	PVSelectionHandleItem(handle_type type,
	                      PVSelectionRectangleItem* sel_rect);

	/**
	 * set the handle's pen's color
	 *
	 * @param col the color to use
	 */
	void set_pen_color(QColor col);

	/**
	 * set the handle's brush's color
	 *
	 * @param col the color to use
	 */
	void set_brush_color(QColor col);

	/**
	 * change the scale factors to use
	 *
	 * As QGraphicsItem::ItemIgnoresTransformations affects all coordinates,
	 * it can not be used for any handles. This implies moreover a slowdown
	 * when moving an handle in a really wide viewport.
	 *
	 * @param xscale the scale factor along X
	 * @param yscale the scale factor along Y
	 */
	void set_scale(const qreal xscale, const qreal yscale);

	/**
	 * get the x scale factor
	 *
	 * @return the x scale factor
	 */
	qreal get_x_scale() const
	{
		return _xscale;
	}

	/**
	 * get the y scale factor
	 *
	 * @return the y scale factor
	 */
	qreal get_y_scale() const
	{
		return _yscale;
	}

public:
	/**
	 * get the handle's bounding box
	 *
	 * @reimpl QGraphicsItem::boundingRect()
	 *
	 * @return the handle's bounding box
	 */
	QRectF boundingRect() const override;

	/**
	 * paint the handle
	 *
	 * @reimpl QGraphicsItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
	 *
	 * @param painter the current painter
	 * @param option the current style
	 * @param widget the current QWidget
	 */
	void paint(QPainter* painter, const QStyleOptionGraphicsItem* option,
	           QWidget* widget) override;

protected:
	/**
	 * highlight the handle.
	 *
	 * @reimpl QGraphicsItem::hoverEnterEvent(QGraphicsSceneHoverEvent* event)
	 *
	 * @param event the hover event
	 */
	void hoverEnterEvent(QGraphicsSceneHoverEvent* event) override;

	/**
	 * unhighlight the handle.
	 *
	 * @reimpl QGraphicsItem::hoverLeaveEvent(QGraphicsSceneHoverEvent* event)
	 *
	 * @param event the hover event
	 */
	void hoverLeaveEvent(QGraphicsSceneHoverEvent* event) override;

	/**
	 * initiate a handle movement
	 *
	 * @reimpl QGraphicsItem::mousePressEvent(QGraphicsSceneMouseEvent* event)
	 *
	 * @param event the hover event
	 */
	void mousePressEvent(QGraphicsSceneMouseEvent* event) override;

	/**
	 * terminate a handle movement
	 *
	 * @reimpl QGraphicsItem::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
	 *
	 * @param event the hover event
	 */
	void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;

	/**
	 * process a handle movement
	 *
	 * @reimpl QGraphicsItem::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
	 *
	 * @param event the hover event
	 */
	void mouseMoveEvent(QGraphicsSceneMouseEvent* event) override;

protected:
	/**
	 * get the associated selection rectangle
	 *
	 * @return the associated selection rectangle
	 */
	PVSelectionRectangleItem* get_selection_rectangle();

	/**
	 * get the associated selection rectangle (const version)
	 *
	 * @return the associated selection rectangle
	 */
	const PVSelectionRectangleItem* get_selection_rectangle() const;

	/**
	 * update the associated selection rectangle's geometry
	 */
	void update_selection_rectangle_geometry(const QPointF& p);

	/**
	 * update the handle geometry according to a rectangle
	 *
	 * @param rect the associated selection rectangle
	 */
	void update_geometry(const QRectF& rect);

	/**
	 * swap horizontally the handle type
	 */
	void swap_horizontally();

	/**
	 * swap vertically the handle type
	 */
	void swap_vertically();

	/**
	 * set if the handle is visible or not
	 *
	 * Contrary to QGraphicsItem::setVisible which is not overloadable,
	 * this one is only used to not paint the handle
	 */
	void set_visible(bool visible);

	/**
	 * tell if the handle is visible or not
	 *
	 * @return true if the handle is visible; false otherwise
	 */
	bool is_visible() const;

	/**
	 * activate or not the handle's cursor
	 *
	 * This is needed to keep the current cursor when the selection rectangle
	 * is created using begin/step/end or when a handle is moved.
	 *
	 * @param use_own is true when the cursor must be set according to the
	 * handle's type, false otherwise to set the default mouse cursor
	 */
	void activate_cursor(bool use_own = true);

	/**
	 * set if the handle must be forced as hidden or not
	 *
	 * @param hidden true if must always be hidden, false otherwise
	 */
	void force_hidden(bool hidden);

	/**
	 * tell if the handle's type matches a mask or not
	 *
	 * A handle typed CENTER does not match any mask.
	 *
	 * @param mask the mask to test the handle's type.
	 *
	 * @return true if the handle's types matches mask or not
	 */
	bool is_type(int mask);

private:
	PVSelectionRectangleItem* _sel_rect;
	QPen                      _pen;
	QBrush                    _brush;
	QColor                    _brush_color;
	QRectF                    _rect;
	QPointF                   _ref;
	qreal                     _xscale;
	qreal                     _yscale;
	int                       _type;
	bool                      _is_visible;
	bool                      _always_hidden;
};

}

#endif // PVWIDGETS_PVSELECTIONHANDLEITEM_H
