
#ifndef PVWIDGETS_PVGRAPHICSVIEW_H
#define PVWIDGETS_PVGRAPHICSVIEW_H

#include <QWidget>
#include <QTransform>
#include <QEvent>

#include <vector>
#include <list>
#include <map>

#include <pvkernel/widgets/PVGraphicsViewInteractor.h>

class QScrollBar64;

class QGridLayout;
class QGraphicsScene;
class QPainter;
class QPaintEvent;
class QResizeEvent;
class QGLFormat;

namespace PVWidgets {

class PVGraphicsViewInteractorScene;

namespace __impl {
class PVViewportEventFilter;
}

/**
 * @class PVGraphicsView
 *
 * @brief a widget which mimics QGraphicsView but which uses QScrollbar64.
 *
 * This widget reproduces QGraphicsView's behaviours used in Picviz Inspector.
 * So that, the differences are:
 * - the members functions name have been adapted to Picviz Labs coding style
 * - no caching when rendering (the repainted area is always renderer from
 *   scratch)
 * - margins can be defined in viewport space to have free space around the
 *   area used to render the scene.
 * - no added border when computing the transformation matrix to make the
 *   scene enter in the viewport. So that the rendering differs from
 *   QGraphicsView.
 *
 * Four coordinates system are thus living here:
 * - the scene coordinate system
 * - the margined viewport coordinate system (with margins taken into account)
 * - the view coordinate system (with margins taken into account). This can be
 *   defined as the margined viewport coordinate system translated by the scrollbar
 *   values.
 * - the viewport coordinate system (the full viewport)
 *
 * But the real difference between PVGraphicsView and QGraphicsView is the way
 * events are processed.
 *
 * QGraphicsView uses virtual method and signals/slots to do the job,
 * PVGraphicsView is based on a interactors system. This sytem mimics the QEventFilter
 * approach but it allows to specify the order of the filter (that QEventFilter does
 * not allow).
 *
 * @todo restrict scene events in the display area (i.e. viewport minus the
 * margins)?
 */
class PVGraphicsView : public QWidget
{
	Q_OBJECT

	friend class __impl::PVViewportEventFilter;
	friend class PVGraphicsViewInteractorScene;

	typedef std::vector<PVGraphicsViewInteractorBase*> interactor_enum_t;
	typedef std::list<PVGraphicsViewInteractorBase*> interactor_list_t;
	typedef std::map<QEvent::Type, interactor_list_t > interactor_affectation_map_t;

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

	~PVGraphicsView();

public:
	/**
	 * Returns the viewport.
	 */
	QWidget *get_viewport() const;

	/**
	 * Set the viewport's widget.
	 */
	void set_viewport(QWidget* w);

	/*! \brief Set the viewport's widget as a QGLWidget if possible.
	 * 
	 * This function sets the viewport's widget as a QGLWidget if possible, and
	 * fallback to the current viewport otherwise..
	 *
	 * \return true if a valid QGLWidget could have been created (that is, in
	 * most case, if OpenGL is available on the running system), and false
	 * otherwise. If Qt has been compiled with no OpenGL support, this will
	 * always return false and do nothing.
	 */
	bool set_gl_viewport(QGLFormat const& format);

	/*! \brief Set the viewport's widget as a QGLWidget if possible.
	 * 
	 * This will use a default QGLFormat.
	 *
	 * \note This function is provided so that QGLFormat can be
	 * forward-declarated here, thus providing a stable API even if no OpenGL
	 * support has been built within Qt.
	 */
	bool set_gl_viewport();

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
	 * Maps a point from margined viewport's space to scene's space.
	 *
	 * @param p the point to transform in scene space
	 */
	QPointF map_margined_to_scene(const QPointF &p) const
	{
		return map_to_scene(map_from_margined(p));
	}

	/**
	 * This convenience function is equivalent to calling
	 * map_to_scene(QPointF(@a x, @a y)).
	 */
	QPointF map_margined_to_scene(const qreal x, const qreal y) const
	{
		return map_margined_to_scene(QPointF(x, y));
	}

	/**
	 * Maps a rectangle from margined viewport's space to scene's space.
	 *
	 * @param r the rectangle to map in scene space
	 */
	QRectF map_margined_to_scene(const QRectF &r) const
	{
		return map_to_scene(map_from_margined(r));
	}

	/**
	 * This convenience function is equivalent to calling
	 * map_to_scene(QRectF(@a x, @a y, @a w, @a h)).
	 */
	QRectF map_margined_to_scene(const qreal x, const qreal y,
	                    const qreal w, const qreal h) const
	{
		return map_margined_to_scene(QRectF(x, y, w, h));
	}

	/**
	 * Maps a point from scene's space to margined viewport's space.
	 *
	 * @param r the rectangle to map in scene space
	 */
	QPointF map_margined_from_scene(const QPointF &p) const
	{
		return map_to_margined(map_from_scene(p));
	}

	/**
	 * This convenience function is equivalent to calling
	 * map_from_scene(QPointF(@a x, @a y)).
	 */
	QPointF map_margined_from_scene(const qreal x, const qreal y) const
	{
		return map_margined_from_scene(QPointF(x, y));
	}

	/**
	 * Maps a rectangle from scene's space to margined viewport's space.
	 *
	 * @param r the rectangle to map in the scene space
	 */
	QRectF map_margined_from_scene(const QRectF &r) const
	{
		return map_to_margined(map_from_scene(r));
	}

	/**
	 * This convenience function is equivalent to calling
	 * map_from_scene(QRectF(@a x, @a y, @a w, @a h)).
	 */
	QRectF map_margined_from_scene(const qreal x, const qreal y,
	                    const qreal w, const qreal h) const
	{
		return map_margined_from_scene(QRectF(x, y, w, h));
	}

public:
	/**
	 * Maps a point from viewport's space to scene's space.
	 *
	 * @param p the point to transform in scene space
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
	 * Maps a rectangle from margined viewport's space to scene's space.
	 *
	 * @param r the rectangle to map in scene space
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
	 * Maps a point from scene's space to margined viewport's space.
	 *
	 * @param r the rectangle to map in scene space
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
	 * Maps a rectangle from scene's space to margined viewport's space.
	 *
	 * @param r the rectangle to map in the scene space
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
	 * Maps a rectange from margined viewport's space to viewport's space.
	 *
	 * @param r Rectangle in the margined view coordinate system to map
	 */
	QRectF map_from_margined(QRectF const& r) const;

	/**
	 * Maps a point from margined viewport's space to viewport's space.
	 *
	 * @param p Point in the margined view coordinate system to map
	 */
	QPointF map_from_margined(QPointF const& p) const;

	/**
	 * Maps a rectange to margined viewport's space from viewport's space.
	 *
	 * @param r Rectangle in the viewport coordinate system to map
	 */
	QRectF map_to_margined(QRectF const& r) const;

	/**
	 * Maps a point to margined viewport's space from viewport's space.
	 *
	 * @param p Point in the viewport coordinate system to map
	 */
	QPointF map_to_margined(QPointF const& p) const;

	/**
	 * Returns the transformation that maps the unmargined viewport to the margined viewport.
	 */
	QTransform get_transform_to_margined_viewport() const;

	/**
	 * Returns the transformation that maps the margined viewport to the unmargined viewport.
	 */
	QTransform get_transform_from_margined_viewport() const;

public:
	/**
	 * Sets the scene's visible area.
	 *
	 * This method configures the scene's area the view can display. To
	 * reset the visible area to the entire scene rectangle, use a null
	 * rectangle as parameter.
	 *
	 * @param r the visible area in scene space
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
	 * Sets a transformation (scene to view)
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
	 * Centers the view on a given scene position.
	 *
	 * This method simply translate the view.
	 *
	 * \param pos Position in scene coordinate system on which the view must be centered
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
	 * Return the margined viewport's height in which the scene is rendered.
	 *
	 * This value depends on scene's margins.
	 *
	 * @return the margined viewport height used to render the scene
	 */
	int get_margined_viewport_height() const
	{
		return _viewport->rect().height() - (_scene_margin_top + _scene_margin_bottom);
	}

	/**
	 * Return the margined viewport's width in which the scene is rendered.
	 *
	 * This value depends on scene's margins.
	 *
	 * @return the margined viewport width used to render the scene
	 */
	int get_margined_viewport_width() const
	{
		return _viewport->rect().width() - (_scene_margin_left + _scene_margin_right);
	}

	/**
	 * Returns the effective area used to display the scene in viewport coordinate system.
	 *
	 * This value depends on scene's margins, and uses the viewport coordinate system.
	 *
	 * @return the effective drawing area rectangle in viewport space.
	 */
	QRect get_margined_viewport_rect() const
	{
		return QRect(_scene_margin_left, _scene_margin_top,
		             get_margined_viewport_width(), get_margined_viewport_height());
	}

	/**
	 * Returns scene rectangle that is visible (and will be rendered)
	 *
	 * @return The scene rectange that is visible, in scene space.
	 */
	QRectF get_visible_scene_rect() const;

	/**
	 * Returns the current viewport widget.
	 */
	inline QWidget* get_viewport()
	{
		return _viewport;
	}

	/**
	 * Set scene alignment when it fits in viewport.
	 *
	 * @param the ored values for horizontal and vertical alignment mode
	 */
	void set_alignment(const Qt::Alignment align);

	/**
	 * Returns the scene alignment mode when it fits in viewport.
	 *
	 * @return the used ored values for horizontal and vertical alignment mode
	 */
	Qt::Alignment get_alignment() const
	{
		return _alignment;
	}

protected:
	/**
	 * declare a new interactor in the instance context
	 *
	 * @param params the parameter list to pass to the constructor
	 *
	 * @return a pointer on the declared interactor
	 */
	template <typename T, typename... P>
	T* declare_interactor(P && ... params)
	{
		T* new_interactor = new T(this, std::forward<P>(params)...);
		_interactor_enum.push_back(new_interactor);
		return new_interactor;
	}

	/**
	 *  Remove a previously installed interactor
	 *
	 * the interactor is implictly unregistered.
	 *
	 * @param interactor [in] The pointer to the interactor to remove (previously returned by install_interactor).
	 *
	 * @note the interactor is not freed.
	 *
	 * @sa declare_interactor
	 */
	void undeclare_interactor(PVGraphicsViewInteractorBase* interactor);

	/**
	 * register an interactor as the first to process for a given QEvent:Type
	 *
	 * @param type the event's type to attach an interactor to
	 * @param interactor the interactor to attach
	 */
	void register_front_one(QEvent::Type type, PVGraphicsViewInteractorBase* interactor);

	/**
	 * register an interactor as the first to process for all supported QEvent::Type
	 *
	 * @param type the event's type to attach an interactor to
	 * @param interactor the interactor to attach
	 */
	void register_front_all(PVGraphicsViewInteractorBase* interactor);

	/**
	 * register an interactor as the last to process for a given QEvent:Type
	 *
	 * @param type the event's type to attach an interactor to
	 * @param interactor the interactor to attach
	 */
	void register_back_one(QEvent::Type type, PVGraphicsViewInteractorBase* interactor);

	/**
	 * register an interactor as the last to process for all supported QEvent::Type
	 *
	 * @param type the event's type to attach an interactor to
	 * @param interactor the interactor to attach
	 */
	void register_back_all(PVGraphicsViewInteractorBase* interactor);

	/**
	 * unregister an interactor for a given QEvent:Type
	 *
	 * @param type the event's type to detach an interactor to
	 * @param interactor the interactor to detach
	 */
	void unregister_one(QEvent::Type type, PVGraphicsViewInteractorBase* interactor);

	/**
	 * unregister an interactor for a all supported QEvent:Type
	 *
	 * @param interactor the interactor to detach
	 */
	void unregister_all(PVGraphicsViewInteractorBase* interactor);

	/**
	 * tell if a QEvent type is supported or not
	 */
	static bool is_event_supported(QEvent::Type type);

	/**
	 * process the event according to its interactors list
	 *
	 * @param event the QEvent to process
	 */
	bool call_interactor(QEvent *event);

	/**
	 * install a default PVGraphicsViewInteractorScene
	 *
	 * i.e.: all events are registered last except for mouse's ones which
	 * are registered first.
	 *
	 * @note This method is not implictly called.
	 */
	void install_default_scene_interactor();

protected:
	/**
	 * Redraws the widget according to the paint event.
	 *
	 * @param event the corresponding paint event
	 */
	//virtual void paintEvent(QPaintEvent *event) override;

	/**
	 * Resizes the widget according to the resize event.
	 *
	 * @param event the corresponding resize event
	 */
	virtual void resizeEvent(QResizeEvent *event) override;

protected:
	// Called by PVViewportEventFilter
	bool viewportPaintEvent(QPaintEvent* event);

protected:
	void contextMenuEvent(QContextMenuEvent *event) override;

	bool event(QEvent *event) override;

	void focusInEvent(QFocusEvent *event) override;
	// does not need reimplementation
	// bool focusNextPrevChild(bool next) override;
	void focusOutEvent(QFocusEvent *event) override;

	void keyPressEvent(QKeyEvent *event) override;
	void keyReleaseEvent(QKeyEvent *event) override;

	void mouseDoubleClickEvent(QMouseEvent *event) override;
	void mouseMoveEvent(QMouseEvent *event) override;
	void mousePressEvent(QMouseEvent *event) override;
	void mouseReleaseEvent(QMouseEvent *event) override;

	void wheelEvent(QWheelEvent *event) override;

#if 0
	// needed or not?
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
	 * The painter is set so that it uses the margined viewport coordinate system.
	 *
	 * @param painter the used painter
	 * @param margined_rect the area to redraw, in the margined viewport coordinate system.
	 */
	virtual void drawBackground(QPainter *painter, const QRectF &margined_rect);

	/**
	 * Draws scene's foreground.
	 *
	 * The painter is set so that it uses the margined viewport coordinate system.
	 *
	 * @param painter the used painter
	 * @param margined_rect the area to redraw, in the margined viewport coordinate system.
	 */
	virtual void drawForeground(QPainter *painter, const QRectF &margined_rect);

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
	 * Recomputes the margins if necessary. This function can safely call recompute_viewport.
	 */
	virtual void recompute_margins();

	/**
	 * Recomputes the viewport geometry and update scrollbars visibility
	 * if needed.
	 */
	void recompute_viewport();

	/**
	 * Computes the new value of _screen_offset_x
	 *
	 * It depends on the horizontal aligment mode and the scene margins.
	 *
	 * @param view_width the width of the viewport area used to render the scene
	 * @param scene_rect the scene rectangle projected into the screen coordinates
	 * system.
	 *
	 * @return the new _screen_offset_x value
	 */
	qreal compute_screen_offset_x(const qint64 view_width, const QRectF scene_rect) const;

	/**
	 * Computes the new value of _screen_offset_y
	 *
	 * It depends on the vertical aligment mode and the scene margins.
	 *
	 * @param view_height the height of the viewport area used to render the scene
	 * @param scene_rect the scene rectangle projected into the screen coordinates
	 * system.
	 *
	 * @return the new _screen_offset_y value
	 */
	qreal compute_screen_offset_y(const qint64 view_height, const QRectF scene_rect) const;

	/**
	 * Recenter the view according to the given anchor
	 *
	 * @param anchor the anchor mode to use.
	 */
	void center_view(ViewportAnchor anchor);

	/**
	 * Returns the horizontal offset used to center the scene in the viewport
	 * when the first is entirely contained in the second.
	 */
	qreal get_scroll_x() const;

	/**
	 * Returns the vertical offset used to center the scene in the viewport
	 * when the first is entirely contained in the second.
	 */
	qreal get_scroll_y() const;

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

	Qt::Alignment       _alignment;

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

	__impl::PVViewportEventFilter* _viewport_event_filter;

	interactor_enum_t            _interactor_enum;
	interactor_affectation_map_t _interactor_map;

private:
	static QEvent::Type  _usable_events[];
	static QEvent::Type *_usable_events_end;
};

}

#endif // PVWIDGETS_PVGRAPHICSVIEW_H
