
#ifndef PVWIDGETS_PVGRAPHICSVIEW_H
#define PVWIDGETS_PVGRAPHICSVIEW_H

#include <QWidget>
#include <QTransform>

class QScrollBar64;

class QGridLayout;
class QGraphicsScene;
class QPainter;
class QPaintEvent;
class QResizeEvent;

namespace PVWidgets {

/**
 * @class PVGraphicsView
 *
 * @brief a widget which mimics QGraphicsView but which uses QScrollbar64.
 *
 * This widget reproduces QGraphicsView's behaviours used in Picviz Inspector
 *
 * The differences are:
 * - the members functions name have been adapted to Picviz Labs coding style
 * - no caching when rendering (the repainted area is always renderer from
 *   scratch)
 * - no support for alignment
 * - no added border when computing the transformation matrix to make the
 *   scene enter in the viewport. So that the rendering differs from
 *   QGraphicsView.
 */
class PVGraphicsView : public QWidget
{
public:
	typedef enum {
		NoAnchor,
		AnchorViewCenter,
		AnchorUnderMouse
	} ViewportAnchor;

public:
	PVGraphicsView(QWidget *parent = nullptr);

	PVGraphicsView(QGraphicsScene *scene = nullptr,
	               QWidget *parent = nullptr);

public:
	/**
	 * Returns the viewport.
	 */
	QWidget *get_viewport() const;

public:
	/**
	 * Sets displayed scene.
	 *
	 * @param scene the new used QGraphicsScene
	 */
	void set_scene(QGraphicsScene *scene);

	/**
	 * Returns the displayed scene.
	 */
	QGraphicsScene *get_scene()
	{
		return _scene;
	}

	/**
	 * Returns the displayed scene.
	 */
	const QGraphicsScene *get_scene() const
	{
		return _scene;
	}

public:
	/**
	 * Maps a point from view's space to scene's space.
	 *
	 * @param p the point to transform
	 */
	QPointF map_to_scene(const QPointF &p) const;

	/**
	 * This convenience function is equivalent to calling
	 * map_to_scene(QPointF(@a x, @a y)).
	 */
	QPointF map_to_scene(const qreal x, const qreal y) const
	{
		return map_to_scene(QPointF(x, y));
	}

	/**
	 * Maps a rectangle from view's space to scene's space.
	 *
	 * @param r the rectangle to map
	 */
	QRectF map_to_scene(const QRectF &r) const;

	/**
	 * This convenience function is equivalent to calling
	 * map_to_scene(QRectF(@a x, @a y, @a w, @a h)).
	 */
	QRectF map_to_scene(const qreal x, const qreal y,
	                    const qreal w, const qreal h) const
	{
		return map_to_scene(QRectF(x, y, w, h));
	}

	/**
	 * Maps a point from scene's space to view's space.
	 *
	 * @param r the rectangle to map
	 */
	QPointF map_from_scene(const QPointF &p) const;

	/**
	 * This convenience function is equivalent to calling
	 * map_from_scene(QPointF(@a x, @a y)).
	 */
	QPointF map_from_scene(const qreal x, const qreal y) const
	{
		return map_from_scene(QPointF(x, y));
	}

	/**
	 * Maps a rectangle from scene's space to view's space.
	 *
	 * @param r the rectangle to map
	 */
	QRectF map_from_scene(const QRectF &r) const;

	/**
	 * This convenience function is equivalent to calling
	 * map_from_scene(QRectF(@a x, @a y, @a w, @a h)).
	 */
	QRectF map_from_scene(const qreal x, const qreal y,
	                    const qreal w, const qreal h) const
	{
		return map_from_scene(QRectF(x, y, w, h));
	}

public:
	/**
	 * Sets the scene's visible area.
	 *
	 * This method configures the scene's area the view can display. To
	 * reset the visible area to the entire scene rectangle, use a null
	 * rectangle as parameter.
	 *
	 * @param r the visible area
	 */
	void set_scene_rect(const QRectF &r);

	/**
	 * This convenience function is equivalent to calling
	 * set_scene_rect(QRectF(@a x, @a y, @a w, @a h)).
	 */
	void set_scene_rect(const qreal x, const qreal y, const qreal w, const qreal h)
	{
		set_scene_rect(QRectF(x, y, w, h));
	}

	/**
	 * Return the scene's visible area.
	 */
	QRectF get_scene_rect() const;

	/**
	 * Sets a transformation (scene to screen)
	 *
	 * @param t the transformation
	 * @param combine a flag telling if @a t is combined with the current
	 * transformation or not
	 */
	void set_transform(const QTransform &t, bool combine = false);

	/**
	 * Makes the visible area fit in the view.
	 *
	 * @param mode an indication on how the area must fits in window
	 *
	 * @bug not stable: 2 consecutive calls with the same parameter can
	 * have a visual effect due to the appearance/disappearance of
	 * scrollbars.
	 */
	void fit_in_view(Qt::AspectRatioMode mode = Qt::KeepAspectRatio);

	/**
	 * Centers the view on a given position.
	 *
	 * This method simply translate the view.
	 */
	void center_on(const QPointF &pos);

	/**
	 * This convenience function is equivalent to calling
	 * set_scene_rect(QRectF(@a x, @a y, @a w, @a h)).
	 */
	void center_on(const qreal x, const qreal y)
	{
		center_on(QPointF(x, y));
	}

public:
	QScrollBar64 *get_horizontal_scrollbar() const
	{
		return _hbar;
	}

	QScrollBar64 *get_vertical_scrollbar() const
	{
		return _vbar;
	}

	/**
	 * Sets the policy for the horizontal scrollbar.
	 *
	 * @param policy the policy
	 *
	 * @bug not stable: 2 consecutive calls with the same parameter can
	 * have a visual effect due to the appearance/disappearance of
	 * scrollbars.
	 */
	void set_horizontal_scrollbar_policy(Qt::ScrollBarPolicy policy);

	/**
	 * Returns the policy of the horizontal scrollbar.
	 */
	Qt::ScrollBarPolicy get_horizontal_scrollbar_policy() const;

	/**
	 * Sets the policy for the vertical scrollbar.
	 *
	 * @param policy the policy
	 *
	 * @bug not stable: 2 consecutive calls with the same parameter can
	 * have a visual effect due to the appearance/disappearance of
	 * scrollbars.
	 */
	void set_vertical_scrollbar_policy(Qt::ScrollBarPolicy policy);

	/**
	 * Returns the policy of the vertical scrollbar.
	 */
	Qt::ScrollBarPolicy get_vertical_scrollbar_policy() const;

	/**
	 * Sets how the scene is put back in the viewport when a resize event
	 * occurs.
	 *
	 * @note the value @a AnchorUnderMouse is not supported.
	 *
	 * @param anchor the behaviour to use
	 *
	 * @bug not stable: 2 consecutive calls with the same parameter can
	 * have a visual effect due to the appearance/disappearance of
	 * scrollbars.
	 */
	void set_resize_anchor(const ViewportAnchor anchor);

	/**
	 * Returns the anchor's type used when a resize event occurs.
	 */
	ViewportAnchor get_resize_anchor() const;


	void set_transformation_anchor(const ViewportAnchor anchor);

	/**
	 * Returns the anchor's type used when a transformation change
	 * occurs.
	 */
	ViewportAnchor get_transformation_anchor() const;

	/**
	 * Set the scene's margins
	 *
	 * Defining margins reduces the scene rendering area to have space to
	 * display something else around it.
	 *
	 * @ param left the left margin width
	 * @ param right the right margin width
	 * @ param top the top margin height
	 * @ param bottom the bottom margin height
	 */
	void set_scene_margins(const int left, const int right,
	                       const int top, const int bottom);

	/**
	 * Returns the scene's left margin value.
	 */
	int get_scene_left_margin() const
	{
		return _scene_margin_left;
	}

	/**
	 * Returns the scene's right margin value.
	 */
	int get_scene_right_margin() const
	{
		return _scene_margin_right;
	}

	/**
	 * Returns the scene's top margin value.
	 */
	int get_scene_top_margin() const
	{
		return _scene_margin_top;
	}

	/**
	 * Returns the scene's left margin value.
	 */
	int get_scene_bottom_margin() const
	{
		return _scene_margin_bottom;
	}

	/**
	 * Return the viewport's height in which the scene is rendered.
	 *
	 * This value depends on scene's margins.
	 *
	 * @return the viewport height used to render the scene
	 */
	int get_real_viewport_height() const
	{
		return _viewport->rect().height() - (_scene_margin_top + _scene_margin_bottom);
	}

	/**
	 * Return the viewport's width in which the scene is rendered.
	 *
	 * This value depends on scene's margins.
	 *
	 * @return the viewport width used to render the scene
	 */
	int get_real_viewport_width() const
	{
		return _viewport->rect().width() - (_scene_margin_left + _scene_margin_right);
	}

protected:
	/**
	 * Redraws the widget according to the paint event.
	 *
	 * @param event the corresponding paint event
	 */
	virtual void paintEvent(QPaintEvent *event);

	/**
	 * Resizes the widget according to the resize event.
	 *
	 * @param event the corresponding resize event
	 */
	virtual void resizeEvent(QResizeEvent *event);

protected:
	void contextMenuEvent(QContextMenuEvent *event);

	bool event(QEvent *event);

	void focusInEvent(QFocusEvent *event);
	// bool focusNextPrevChild(bool next); does not need reimplementation
	void focusOutEvent(QFocusEvent *event);

	void keyPressEvent(QKeyEvent *event);
	void keyReleaseEvent(QKeyEvent *event);

	void mouseDoubleClickEvent(QMouseEvent *event);
	void mouseMoveEvent(QMouseEvent *event);
	void mousePressEvent(QMouseEvent *event);
	void mouseReleaseEvent(QMouseEvent *event);

	void wheelEvent(QWheelEvent *event);

#if 0
	virtual void dragEnterEvent(QDragEnterEvent *event);
	virtual void dragLeaveEvent(QDragLeaveEvent *event);
	virtual void dragMoveEvent(QDragMoveEvent *event);
	virtual void dropEvent(QDropEvent *event);
	virtual void inputMethodEvent(QInputMethodEvent *event);
	virtual void showEvent(QShowEvent *event);
	virtual bool viewportEvent(QEvent *event);

	virtual void scrollContentsBy(int dx, int dy);
#endif

protected:
	/**
	 * Draws scene's background.
	 *
	 * @param painter the used painter
	 * @param rect the area to redraw
	 */
	virtual void drawBackground(QPainter *painter, const QRectF &rect);

	/**
	 * Draws scene's foreground.
	 *
	 * @param painter the used painter
	 * @param rect the area to redraw
	 */
	virtual void drawForeground(QPainter *painter, const QRectF &rect);

	/**
	 * \reimpl
	 */
	QSize sizeHint() const;

protected:
	/**
	 * Modifies the viewport..
	 *
	 * @param area a rectangle in scene's coordinates system
	 * @param mode an indication on how the area must fits in window
	 */
	void set_view(const QRectF &area,
	              Qt::AspectRatioMode mode = Qt::KeepAspectRatio);

private:
	/**
	 * Initializes internal stuff.
	 */
	void init();

	/**
	 * Recomputes the viewport geometry and update scrollbars visibility
	 * if needed.
	 */
	void recompute_viewport();

	/**
	 */
	void center_view(ViewportAnchor anchor);

	/**
	 * Returns the horizontal offset used to center the scene in the viewport
	 * when the first is entirely contained in the second.
	 */
	const qreal get_scroll_x() const;

	/**
	 * Returns the vertical offset used to center the scene in the viewport
	 * when the first is entirely contained in the second.
	 */
	const qreal get_scroll_y() const;

	/**
	 * Returns the offset vector used to center the scene in the viewport
	 * when the first is entirely contained in the second.
	 */
	const QPointF get_scroll() const;

private:
	QGridLayout        *_layout;
	QScrollBar64       *_hbar;
	QScrollBar64       *_vbar;
	QWidget            *_viewport;

	QGraphicsScene     *_scene;
	QRectF              _scene_rect;

	Qt::ScrollBarPolicy _hbar_policy;
	Qt::ScrollBarPolicy _vbar_policy;
	ViewportAnchor      _resize_anchor;
	ViewportAnchor      _transformation_anchor;

	int                 _scene_margin_left;
	int                 _scene_margin_right;
	int                 _scene_margin_top;
	int                 _scene_margin_bottom;

	QPointF             _scene_offset;
	qreal               _screen_offset_x;
	qreal               _screen_offset_y;
	QTransform          _transform;
	QTransform          _inv_transform;

	Qt::MouseButton     _mouse_pressed_button;
	QPoint              _mouse_pressed_screen_coord;
	QPoint              _mouse_pressed_view_coord;
	QPointF             _mouse_pressed_scene_coord;

	QPoint              _last_mouse_move_screen_coord;
	QPointF             _last_mouse_move_scene_coord;

	QPointF             _last_center_coord;
};

}

#endif // PVWIDGETS_PVGRAPHICSVIEW_H
