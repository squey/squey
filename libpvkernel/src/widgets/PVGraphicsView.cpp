#include <pvkernel/widgets/PVGraphicsView.h>
#include <pvkernel/widgets/PVGraphicsViewInteractor.h>
#include <pvkernel/widgets/PVGraphicsViewInteractorScene.h>

#include <QGridLayout>
#include <QGraphicsScene>
#include <QGraphicsItem>
#include <QPaintEvent>
#include <QPainter>
#include <QApplication>
#include <QEvent>
#include <QGraphicsSceneWheelEvent>
#include <QScrollBar64>
#include <QDebug>

#ifndef QT_NO_OPENGL
#include <QGLWidget>
#endif

// to mimic QGraphicsView::::sizeHint() :-D
#include <QApplication>
#include <QDesktopWidget>

#include <iostream>
#include <cassert>
#include <algorithm>

static inline qint64 sb_round(const qreal &d)
{
	if (d <= (qreal) INT64_MIN)
		return INT64_MIN;
	else if (d >= (qreal) INT64_MAX)
		return INT64_MAX;
	return (d > (qreal)0.0) ? floor(d + (qreal)0.5) : ceil(d - (qreal)0.5);
}

#define print_r(R) print_rect(R)
#define print_rect(R) __print_rect(#R, R)

template <typename R>
void __print_rect(const char *text, const R &r)
{
	std::cout << text << ": "
	          << r.x() << " " << r.y() << ", "
	          << r.width() << " " << r.height()
	          << std::endl;
}

#define print_p(P) print_point(P)
#define print_point(P) __print_point(#P, P)

template <typename P>
void __print_point(const char *text, const P &p)
{
	std::cout << text << ": "
	          << p.x() << " " << p.y()
	          << std::endl;
}

#define print_t(T) print_transform(T)
#define print_transform(T) __print_transform(#T, T)

template <typename T>
void __print_transform(const char *text, const T &t)
{
	std::cout << text << ": " << std::endl
	          << t.m11() << " " << t.m21() << " " << t.m31() << std::endl
	          << t.m12() << " " << t.m22() << " " << t.m32() << std::endl
	          << t.m13() << " " << t.m23() << " " << t.m33() << std::endl;
}

namespace PVWidgets { namespace __impl {

class PVViewportEventFilter: public QObject
{
public:
	PVViewportEventFilter(PVWidgets::PVGraphicsView* view):
		_view(view)
	{ }

protected:
	bool eventFilter(QObject* obj, QEvent* event)
	{
		if (obj != _view->get_viewport()) {
			return QObject::eventFilter(obj, event);
		}

		switch(event->type()) {
		case QEvent::Paint:
			return _view->viewportPaintEvent(static_cast<QPaintEvent*>(event));
		case QEvent::MouseMove:
			static_cast<QWidget*>(obj)->update();
			return false;
		default:
			break;
		}
		return QObject::eventFilter(obj, event);
	}

private:
	PVWidgets::PVGraphicsView* _view;
};

} }

/*****************************************************************************
 * PVWidgets::PVGraphicsView::_usable_events
 *****************************************************************************/

QEvent::Type PVWidgets::PVGraphicsView::_usable_events[] = {
	QEvent::MouseButtonDblClick,
	QEvent::MouseButtonPress, QEvent::MouseButtonRelease,
	QEvent::MouseMove,
	QEvent::Wheel,
	QEvent::KeyPress, QEvent::KeyRelease,
	QEvent::Resize
};

QEvent::Type* PVWidgets::PVGraphicsView::_usable_events_end = PVWidgets::PVGraphicsView::_usable_events + (sizeof(PVWidgets::PVGraphicsView::_usable_events) / sizeof(QEvent::Type));

/*****************************************************************************
 * PVWidgets::PVGraphicsView::PVGraphicsView
 *****************************************************************************/

PVWidgets::PVGraphicsView::PVGraphicsView(QWidget *parent)
	: QWidget(parent), _viewport(nullptr), _scene(nullptr)
{
	init();
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::PVGraphicsView
 *****************************************************************************/

PVWidgets::PVGraphicsView::PVGraphicsView(QGraphicsScene *scene, QWidget *parent)
	: QWidget(parent), _viewport(nullptr), _scene(nullptr)
{
	init();
	set_scene(scene);
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::~PVGraphicsView
 *****************************************************************************/

PVWidgets::PVGraphicsView::~PVGraphicsView()
{}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::get_viewport
 *****************************************************************************/

QWidget *PVWidgets::PVGraphicsView::get_viewport() const
{
	return _viewport;
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::set_scene
 *****************************************************************************/

void PVWidgets::PVGraphicsView::set_scene(QGraphicsScene *scene)
{
	if(_scene && hasFocus()) {
            _scene->clearFocus();
	}

	_scene = scene;

	if (_scene == nullptr) {
		return;
	}

	_scene->setDefaultViewTransform(get_transform());

	recompute_viewport();

	if (hasFocus()) {
		_scene->setFocus();
	}
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::map_to_scene
 *****************************************************************************/

QPointF PVWidgets::PVGraphicsView::map_to_scene(const QPointF &p) const
{
	/*QPointF np = p + get_scroll();
	return _inv_transform.map(np) + _scene_offset;*/
	return get_transform_to_scene().map(p);
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::map_to_scene
 *****************************************************************************/

QRectF PVWidgets::PVGraphicsView::map_to_scene(const QRectF &r) const
{
	/*QRectF tr = r.translated(get_scroll());
	return _inv_transform.mapRect(tr).translated(_scene_offset);*/
	return get_transform_to_scene().mapRect(r);
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::map_from_scene
 *****************************************************************************/

QPointF PVWidgets::PVGraphicsView::map_from_scene(const QPointF &p) const
{
	/*QPointF np = _transform.map(p - _scene_offset);
	np -= get_scroll();
	return np;*/
	return get_transform_from_scene().map(p);
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::map_from_scene
 *****************************************************************************/

QRectF PVWidgets::PVGraphicsView::map_from_scene(const QRectF &r) const
{
	/*QRectF nr = _transform.mapRect(r.translated(-_scene_offset));
	nr.translate(-get_scroll());
	return nr;*/
	return get_transform_from_scene().mapRect(r);
}

QTransform PVWidgets::PVGraphicsView::get_transform_to_scene() const
{
	QTransform trans_scroll;
	trans_scroll.translate(get_scroll_x(), get_scroll_y());

	QTransform trans_scene_offset;
	trans_scene_offset.translate(_scene_offset.x(), _scene_offset.y());

	return trans_scroll * _inv_transform * trans_scene_offset;
}

QTransform PVWidgets::PVGraphicsView::get_transform_from_scene() const
{
	QTransform trans_scroll;
	trans_scroll.translate(-get_scroll_x(), -get_scroll_y());
	QTransform trans_scene_offset;
	trans_scene_offset.translate(-_scene_offset.x(), -_scene_offset.y());

	return trans_scene_offset * _transform * trans_scroll;
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::set_scene_rect
 *****************************************************************************/

void PVWidgets::PVGraphicsView::set_scene_rect(const QRectF &r)
{
	_scene_rect = r;
	recompute_viewport();
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::get_scene_rect
 *****************************************************************************/

QRectF PVWidgets::PVGraphicsView::get_scene_rect() const
{
	if (_scene_rect.isNull() && (get_scene() != nullptr)) {
		return get_scene()->sceneRect();
	} else {
		return _scene_rect;
	}
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::set_transform
 *****************************************************************************/

void PVWidgets::PVGraphicsView::set_transform(const QTransform &t, bool combine)
{
	if (combine) {
		_transform = t * _transform;
	} else {
		_transform = t;
	}

	_inv_transform = _transform.inverted();

	recompute_margins();
	recompute_viewport();

	if (get_scene()) {
		get_scene()->setDefaultViewTransform(get_transform());
	}

	/* if _transformation_anchor is equal to AnchorUnderMouse while the
	 * mouse is not on the view, there is an translation effect due to the
	 * use of QCursor::pos().
	 * So, when the mouse pointer is outside of the view, AnchorUnderMouse
	 * *must not* be used.
	 */
	if (underMouse()) {
		center_view(_transformation_anchor);
	} else {
		center_view(NoAnchor);
	}
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::fit_in_view
 *****************************************************************************/

void PVWidgets::PVGraphicsView::fit_in_view(Qt::AspectRatioMode mode)
{
	if (get_scene() == nullptr) {
		return;
	}

	set_view(get_scene_rect(), mode);
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::center_on
 *****************************************************************************/

void PVWidgets::PVGraphicsView::center_on(const QPointF &pos)
{
	QPointF pos_view = _transform.map(pos - _scene_offset);
	pos_view -= QPointF(get_margined_viewport_width()/2.0, get_margined_viewport_height()/2.0);

	_hbar->setValue(pos_view.x());
	_vbar->setValue(pos_view.y());
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::fake_mouse_move
 *****************************************************************************/

void PVWidgets::PVGraphicsView::fake_mouse_move()
{
	QMouseEvent e((QEvent::MouseMove),
	              mapFromGlobal(QCursor::pos()),
	              Qt::NoButton,
	              Qt::NoButton,
	              Qt::NoModifier);
	QApplication::sendEvent(this , &e);
}


/*****************************************************************************
 * PVWidgets::PVGraphicsView::set_background_color
 *****************************************************************************/

void PVWidgets::PVGraphicsView::set_background_color(const QColor& color)
{
	_background_color = color;
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::set_horizontal_scrollbar_policy
 *****************************************************************************/

void PVWidgets::PVGraphicsView::set_horizontal_scrollbar_policy(Qt::ScrollBarPolicy policy)
{
	_hbar_policy = policy;
	recompute_viewport();
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::get_horizontal_scrollbar_policy
 *****************************************************************************/

Qt::ScrollBarPolicy PVWidgets::PVGraphicsView::get_horizontal_scrollbar_policy() const
{
	return _hbar_policy;
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::set_vertical_scrollbar_policy
 *****************************************************************************/

void PVWidgets::PVGraphicsView::set_vertical_scrollbar_policy(Qt::ScrollBarPolicy policy)
{
	_vbar_policy = policy;
	recompute_viewport();
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::get_vertical_scrollbar_policy
 *****************************************************************************/

Qt::ScrollBarPolicy PVWidgets::PVGraphicsView::get_vertical_scrollbar_policy() const
{
	return _vbar_policy;
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::set_resize_anchor
 *****************************************************************************/

void PVWidgets::PVGraphicsView::set_resize_anchor(const PVWidgets::PVGraphicsView::ViewportAnchor anchor)
{
	switch(anchor) {
	case NoAnchor:
	case AnchorViewCenter:
		_resize_anchor = anchor;
		break;
	case AnchorUnderMouse:
		qWarning("anchor AnchorUnderMouse is not supported for resize event");
		break;
	}

	if (anchor == AnchorUnderMouse) {
		_viewport->setMouseTracking(true);
		setMouseTracking(true);
	}
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::get_resize_anchor
 *****************************************************************************/

PVWidgets::PVGraphicsView::ViewportAnchor PVWidgets::PVGraphicsView::get_resize_anchor() const
{
	return _resize_anchor;
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::set_transformation_anchor
 *****************************************************************************/

void PVWidgets::PVGraphicsView::set_transformation_anchor(const PVWidgets::PVGraphicsView::ViewportAnchor anchor)
{
	_transformation_anchor = anchor;

	if (anchor == AnchorUnderMouse) {
		_viewport->setMouseTracking(true);
		setMouseTracking(true);
	}
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::get_transformation_anchor
 *****************************************************************************/

PVWidgets::PVGraphicsView::ViewportAnchor PVWidgets::PVGraphicsView::get_transformation_anchor() const
{
	return _transformation_anchor;
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::set_scene_margins
 *****************************************************************************/

void PVWidgets::PVGraphicsView::set_scene_margins(const int left,
                                                  const int right,
                                                  const int top,
                                                  const int bottom)
{
	if ((_scene_margin_left != left) || (_scene_margin_right != right) || (_scene_margin_top != top) || (_scene_margin_bottom != bottom)) {
		_scene_margin_left = left;
		_scene_margin_right = right;
		_scene_margin_top = top;
		_scene_margin_bottom = bottom;
		recompute_viewport();
	}
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::set_alignment
 *****************************************************************************/

void PVWidgets::PVGraphicsView::set_alignment(const Qt::Alignment align)
{
	if (_alignment != align) {
		_alignment = align;
		recompute_viewport();
	}
}

QRectF PVWidgets::PVGraphicsView::map_from_view(QRectF const& r) const
{
	return get_transform_from_view().mapRect(r);
}

QPointF PVWidgets::PVGraphicsView::map_from_view(QPointF const& p) const
{
	return get_transform_from_view().map(p);
}

QRectF PVWidgets::PVGraphicsView::map_to_view(QRectF const& r) const
{
	return get_transform_to_view().mapRect(r);
}

QPointF PVWidgets::PVGraphicsView::map_to_view(QPointF const& p) const
{
	return get_transform_to_view().map(p);
}

QTransform PVWidgets::PVGraphicsView::get_transform_to_view() const
{
	QTransform ret;
	ret.translate(get_scroll_x(), get_scroll_y());
	return ret;
}

QTransform PVWidgets::PVGraphicsView::get_transform_from_view() const
{
	QTransform ret;
	ret.translate(-get_scroll_x(), -get_scroll_y());
	return ret;
}

/////

QRectF PVWidgets::PVGraphicsView::map_from_margined(QRectF const& r) const
{
	return get_transform_from_margined_viewport().mapRect(r);
}

QPointF PVWidgets::PVGraphicsView::map_from_margined(QPointF const& p) const
{
	return get_transform_from_margined_viewport().map(p);
}

QRectF PVWidgets::PVGraphicsView::map_to_margined(QRectF const& r) const
{
	return get_transform_to_margined_viewport().mapRect(r);
}

QPointF PVWidgets::PVGraphicsView::map_to_margined(QPointF const& p) const
{
	return get_transform_to_margined_viewport().map(p);
}

QTransform PVWidgets::PVGraphicsView::get_transform_to_margined_viewport() const
{
	QTransform ret;
	ret.translate(-_scene_margin_left, -_scene_margin_top);
	return ret;
}

QTransform PVWidgets::PVGraphicsView::get_transform_from_margined_viewport() const
{
	QTransform ret;
	ret.translate(_scene_margin_left, _scene_margin_top);
	return ret;
}

QRectF PVWidgets::PVGraphicsView::get_visible_scene_rect() const
{
	return map_to_scene(QRectF(0, 0, get_margined_viewport_width(), get_margined_viewport_height()));
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::viewportPaintEvent
 *****************************************************************************/

bool PVWidgets::PVGraphicsView::viewportPaintEvent(QPaintEvent *event)
{
	if(get_scene() == nullptr) {
		return false;
	}

	const QRectF unmargined_render_rect = event->rect();
	const QRectF margined_render_rect = map_to_margined(unmargined_render_rect);

	// Benchmark of the following line gives about 0.005ms, which is negligeable against the other renderings..
	const QTransform margined_transform = get_transform_from_margined_viewport();

	QPainter painter;
	painter.begin(get_viewport());
	painter.fillRect(unmargined_render_rect, _background_color);

	painter.setTransform(margined_transform, false);
	drawBackground(&painter, margined_render_rect);

	if (get_scene()) {
		const QRectF unmargined_scene_rect = map_to_scene(unmargined_render_rect);

		painter.setTransform(QTransform(), false);
		get_scene()->render(&painter,
		                    unmargined_render_rect,
		                    unmargined_scene_rect,
		                    Qt::IgnoreAspectRatio);
	}

	painter.setTransform(margined_transform, false);
	drawForeground(&painter, margined_render_rect);

	painter.end();

	return false;
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::resizeEvent
 *****************************************************************************/

void PVWidgets::PVGraphicsView::resizeEvent(QResizeEvent *event)
{
	call_interactor(event);
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::enterEvent
 *****************************************************************************/

void PVWidgets::PVGraphicsView::enterEvent(QEvent*)
{
	setFocus(Qt::MouseFocusReason);
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::leaveEvent
 *****************************************************************************/

void PVWidgets::PVGraphicsView::leaveEvent(QEvent*)
{
	clearFocus();
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::contextMenuEvent
 *****************************************************************************/

void PVWidgets::PVGraphicsView::contextMenuEvent(QContextMenuEvent *event)
{
	_mouse_pressed_screen_coord = event->globalPos();
	_mouse_pressed_view_coord = event->pos();
	_mouse_pressed_scene_coord = map_to_scene(_mouse_pressed_view_coord);

	//_last_mouse_move_screen_coord = _mouse_pressed_screen_coord;
	//_last_mouse_move_scene_coord = _mouse_pressed_scene_coord;

	QGraphicsSceneContextMenuEvent scene_event(QEvent::GraphicsSceneContextMenu);

	scene_event.setScenePos(_mouse_pressed_scene_coord);
	scene_event.setScreenPos(_mouse_pressed_screen_coord);

	scene_event.setModifiers(event->modifiers());
	scene_event.setReason((QGraphicsSceneContextMenuEvent::Reason)(event->reason()));
	scene_event.setWidget(_viewport);
	scene_event.setAccepted(false);

	if (get_scene()) {
		QApplication::sendEvent(get_scene(), &scene_event);
	}

	event->setAccepted(scene_event.isAccepted());
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::focusInEvent
 *****************************************************************************/

void PVWidgets::PVGraphicsView::focusInEvent(QFocusEvent *event)
{
	_mouse_pressed_screen_coord = QCursor::pos();
	_mouse_pressed_view_coord = mapFromGlobal(_mouse_pressed_screen_coord);
	_mouse_pressed_scene_coord = map_to_scene(_mouse_pressed_view_coord);

	QWidget::focusInEvent(event);

	if (get_scene()) {
		QApplication::sendEvent(get_scene(), event);
	}
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::focusOutEvent
 *****************************************************************************/

void PVWidgets::PVGraphicsView::focusOutEvent(QFocusEvent *event)
{
	QWidget::focusOutEvent(event);

	if (get_scene()) {
		QApplication::sendEvent(get_scene(), event);
	}
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::keyPressEvent
 *****************************************************************************/

void PVWidgets::PVGraphicsView::keyPressEvent(QKeyEvent *event)
{
	if (!call_interactor(event)) {
		QWidget::keyPressEvent(event);
	}
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::keyReleaseEvent
 *****************************************************************************/

void PVWidgets::PVGraphicsView::keyReleaseEvent(QKeyEvent *event)
{
	if (!call_interactor(event)) {
		QWidget::keyReleaseEvent(event);
	}
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::mouseDoubleClickEvent
 *****************************************************************************/

void PVWidgets::PVGraphicsView::mouseDoubleClickEvent(QMouseEvent *event)
{
	if (!call_interactor(event)) {
		QWidget::mouseDoubleClickEvent(event);
	}
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::mouseMoveEvent
 *****************************************************************************/

void PVWidgets::PVGraphicsView::mouseMoveEvent(QMouseEvent *event)
{
	if (!call_interactor(event)) {
		QWidget::mouseMoveEvent(event);
	}

	update_viewport_cursor();
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::mousePressEvent
 *****************************************************************************/

void PVWidgets::PVGraphicsView::mousePressEvent(QMouseEvent *event)
{
	if (!call_interactor(event)) {
		QWidget::mousePressEvent(event);
	}

	update_viewport_cursor();
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::mouseReleaseEvent
 *****************************************************************************/

void PVWidgets::PVGraphicsView::mouseReleaseEvent(QMouseEvent *event)
{
	if (!call_interactor(event)) {
		QWidget::mouseReleaseEvent(event);
	}

	update_viewport_cursor();
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::wheelEvent
 *****************************************************************************/

void PVWidgets::PVGraphicsView::wheelEvent(QWheelEvent *event)
{
	if (!call_interactor(event)) {
		QWidget::wheelEvent(event);
	}
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::drawBackground
 *****************************************************************************/

void PVWidgets::PVGraphicsView::drawBackground(QPainter *, const QRectF &)
{}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::drawForeground
 *****************************************************************************/

void PVWidgets::PVGraphicsView::drawForeground(QPainter *, const QRectF &)
{}


/*****************************************************************************
 * PVWidgets::PVGraphicsView::sizehint
 *****************************************************************************/

QSize PVWidgets::PVGraphicsView::sizeHint() const
{
	if (get_scene()) {
		QSizeF s = _transform.mapRect(get_scene_rect()).size();
		return s.boundedTo((3 * QApplication::desktop()->size()) / 4).toSize();
	}
	return QSize(256, 192);
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::set_view
 *****************************************************************************/

void PVWidgets::PVGraphicsView::set_view(const QRectF &area, Qt::AspectRatioMode mode)
{
	/* FIXME: not stable when scrollbar are effectively hidden or shown
	 * by ::recompute_viewport because _viewport's size has changed.
	 */
	QTransform transfo;
	qreal viewport_width = get_margined_viewport_width();
	qreal viewport_height = get_margined_viewport_height();

	qreal x_scale = area.width() / viewport_width;
	qreal y_scale = area.height() / viewport_height;
	qreal tx = area.x();
	qreal ty = area.y();

	switch(mode) {
	case Qt::IgnoreAspectRatio:
		break;
	case Qt::KeepAspectRatio:
		if (x_scale < y_scale) {
			tx -= 0.5 * (viewport_height * y_scale - area.width());
			x_scale = y_scale;
		} else {
			ty -= 0.5 * (viewport_height * x_scale - area.height());
			y_scale = x_scale;
		}
		break;
	case Qt::KeepAspectRatioByExpanding:
		if (x_scale < y_scale) {
			ty -= 0.5 * (viewport_height * x_scale - area.height());
			y_scale = x_scale;
		} else {
			tx -= 0.5 * (viewport_height * y_scale - area.width());
			x_scale = y_scale;
		}
		break;
	}

	_scene_offset = QPointF(tx, ty);
	transfo.scale(1. / x_scale, 1. / y_scale);

	set_transform(transfo);
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::update_viewport_cursor
 *****************************************************************************/

void PVWidgets::PVGraphicsView::update_viewport_cursor()
{
	QPointF p = map_to_scene(mapFromGlobal(QCursor::pos()));
	QGraphicsItem* item = get_scene()->itemAt(p, get_transform());

	if (item && item->hasCursor()) {
		_viewport->setCursor(item->cursor());
	} else {
		_viewport->setCursor(_viewport_cursor);
	}
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::init
 *****************************************************************************/

void PVWidgets::PVGraphicsView::init()
{
	_background_color = Qt::black;

	_hbar_policy = Qt::ScrollBarAsNeeded;
	_vbar_policy = Qt::ScrollBarAsNeeded;
	_resize_anchor = NoAnchor;
	_transformation_anchor = AnchorViewCenter;

	_transform.reset();
	_inv_transform.reset();

	setFocusPolicy(Qt::WheelFocus);
	setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

	_scene_margin_left = 0;
	_scene_margin_right = 0;
	_scene_margin_top = 0;
	_scene_margin_bottom = 0;

	_alignment = Qt::AlignHCenter | Qt::AlignVCenter;

	_layout = new QGridLayout(this);
	_layout->setSpacing(0);
	_layout->setContentsMargins(0, 0, 0, 0);

	setLayout(_layout);

	_viewport_event_filter = new __impl::PVViewportEventFilter(this);

	_hbar = new QScrollBar64(Qt::Horizontal);
	_vbar = new QScrollBar64(Qt::Vertical);

	set_viewport(new QWidget());

	_layout->addWidget(_hbar, 1, 0);
	_layout->addWidget(_vbar, 0, 1);
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::set_viewport
 *****************************************************************************/

void PVWidgets::PVGraphicsView::set_viewport(QWidget* w)
{
	bool mouse_tracking = w->hasMouseTracking();
	if (get_viewport()) {
		mouse_tracking = get_viewport()->hasMouseTracking();
		_layout->removeWidget(_viewport);
		get_viewport()->deleteLater();
	}

	_viewport = w;
	// Redirect any focus and key/mouse event to this widget
	_viewport->setFocusPolicy(Qt::WheelFocus);
	_viewport->setFocusProxy(this);

	// If mouse tracking was enabled, re-enable it !
	_viewport->setMouseTracking(mouse_tracking);

	// Connect hbar and vbar valueChanged events
	connect(_hbar, SIGNAL(valueChanged(qint64)),
	        _viewport, SLOT(update()));
	connect(_vbar, SIGNAL(valueChanged(qint64)),
	        _viewport, SLOT(update()));

	_viewport->installEventFilter(_viewport_event_filter);
	_layout->addWidget(_viewport, 0, 0);

	_viewport->lower();
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::set_gl_viewport
 *****************************************************************************/

bool PVWidgets::PVGraphicsView::set_gl_viewport(QGLFormat const& format)
{
#ifdef QT_NO_OPENGL
	return false;
#else
	QGLWidget* w = new QGLWidget(format);
	if (!w->isValid()) {
		w->deleteLater();
		return false;
	}
	set_viewport(w);
	return true;
#endif
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::set_gl_viewport
 *****************************************************************************/

bool PVWidgets::PVGraphicsView::set_gl_viewport()
{
#ifdef QT_NO_OPENGL
	return false;
#else
	return set_gl_viewport(QGLFormat());
#endif
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::set_viewport_cursor
 *****************************************************************************/

void PVWidgets::PVGraphicsView::set_viewport_cursor(const QCursor& cursor)
{
	get_viewport()->setCursor(cursor);
	_viewport_cursor = cursor;
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::recompute_margins
 *****************************************************************************/
void PVWidgets::PVGraphicsView::recompute_margins()
{
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::recompute_viewport
 *****************************************************************************/

void PVWidgets::PVGraphicsView::recompute_viewport()
{
	qint64 view_width =  get_margined_viewport_width();
	qint64 view_height = get_margined_viewport_height();

	QRectF scene_rect = _transform.mapRect(get_scene_rect().translated(-_scene_offset));

	if (_hbar_policy == Qt::ScrollBarAlwaysOff) {
		_hbar->setRange(0, 0);
		_hbar->setVisible(false);
		_screen_offset_x = compute_screen_offset_x(view_width, scene_rect);
	} else {
		qint64 scene_left = sb_round(scene_rect.left());
		qint64 scene_right = sb_round(scene_rect.right() - view_width);

		if (scene_left >= scene_right) {
			_hbar->setRange(0, 0);
			_hbar->setVisible(_hbar_policy == Qt::ScrollBarAlwaysOn);
			_screen_offset_x = compute_screen_offset_x(view_width, scene_rect);
		} else {
			_hbar->setRange(scene_left, scene_right);
			_hbar->setPageStep(view_width);
			_hbar->setSingleStep(view_width / 20);
			_hbar->setVisible(true);
			_screen_offset_x = _scene_margin_left;
		}
	}

	if (_vbar_policy == Qt::ScrollBarAlwaysOff) {
		_vbar->setRange(0, 0);
		_vbar->setVisible(false);
		_screen_offset_y = compute_screen_offset_y(view_height, scene_rect);
	} else {
		qint64 scene_top = sb_round(scene_rect.top());
		qint64 scene_bottom = sb_round(scene_rect.bottom() - view_height);

		if (scene_top >= scene_bottom) {
			_vbar->setRange(0, 0);
			_vbar->setVisible(_vbar_policy == Qt::ScrollBarAlwaysOn);
			_screen_offset_y = compute_screen_offset_y(view_height, scene_rect);
		} else {
			_vbar->setRange(scene_top, scene_bottom);
			_vbar->setPageStep(view_height);
			_vbar->setSingleStep(view_height / 20);
			_vbar->setVisible(true);
			_screen_offset_y = _scene_margin_top;
		}
	}
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::recompute_screen_offset_x
 *****************************************************************************/

qreal PVWidgets::PVGraphicsView::compute_screen_offset_x(const qint64 view_width,
                                                         const QRectF scene_rect) const
{
	qreal ret = _scene_margin_left;

	switch(_alignment & Qt::AlignHorizontal_Mask) {
	case Qt::AlignLeft:
		break;
	case Qt::AlignRight:
		ret += view_width - (scene_rect.left() + scene_rect.width());
		break;
	case Qt::AlignHCenter:
		ret += 0.5 * (view_width - (scene_rect.left() + scene_rect.right()));
		break;
	}
	return ret;
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::recompute_screen_offset_y
 *****************************************************************************/

qreal PVWidgets::PVGraphicsView::compute_screen_offset_y(const qint64 view_height,
                                                         const QRectF scene_rect) const
{
	qreal ret = _scene_margin_top;

	switch(_alignment & Qt::AlignVertical_Mask) {
	case Qt::AlignTop:
		break;
	case Qt::AlignBottom:
		ret += view_height - (scene_rect.top() + scene_rect.height());
		break;
	case Qt::AlignVCenter:
		ret += 0.5 * (view_height - (scene_rect.top() + scene_rect.bottom()));
		break;
	}
	return ret;
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::center_view
 *****************************************************************************/

void PVWidgets::PVGraphicsView::center_view(ViewportAnchor anchor)
{
	if (anchor == AnchorViewCenter) {
		QPointF p = map_to_scene(get_margined_viewport_rect().center());
		center_on(p);
	} else if (anchor == AnchorUnderMouse) {
		QPointF delta = map_to_scene(get_margined_viewport_rect().center());
		delta -= map_to_scene(get_viewport()->mapFromGlobal(QCursor::pos()));
		center_on(_last_mouse_move_scene_coord + delta);
	} // else if (anchor == NoAnchor) do nothing
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::get_scroll_x
 *****************************************************************************/

qreal PVWidgets::PVGraphicsView::get_scroll_x() const
{
	return _hbar->value() - _screen_offset_x;
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::get_scroll_y
 *****************************************************************************/

qreal PVWidgets::PVGraphicsView::get_scroll_y() const
{
	return _vbar->value() - _screen_offset_y;
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::get_scroll
 *****************************************************************************/

const QPointF PVWidgets::PVGraphicsView::get_scroll() const
{
	return QPointF(get_scroll_x(), get_scroll_y());
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::remove_interactor
 *****************************************************************************/

void PVWidgets::PVGraphicsView::undeclare_interactor(PVGraphicsViewInteractorBase* interactor)
{
	assert(interactor);

	unregister_all(interactor);

	interactor_enum_t::iterator it = std::find(_interactor_enum.begin(),
	                                           _interactor_enum.end(),
	                                           interactor);

	if (it != _interactor_enum.end()) {
		_interactor_enum.erase(it);
	}
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::register_front_one
 *****************************************************************************/

void PVWidgets::PVGraphicsView::register_front_one(QEvent::Type type,
                                                   PVWidgets::PVGraphicsViewInteractorBase* interactor)
{
	assert(is_event_supported(type));
	assert(interactor);

	unregister_one(type, interactor);

	_interactor_map[type].push_front(interactor);
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::register_front_all
 *****************************************************************************/


void PVWidgets::PVGraphicsView::register_front_all(PVWidgets::PVGraphicsViewInteractorBase* interactor)
{
	assert(interactor);

	for (QEvent::Type type : _usable_events) {
		_interactor_map[type].push_front(interactor);
	}
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::register_back_one
 *****************************************************************************/

void PVWidgets::PVGraphicsView::register_back_one(QEvent::Type type,
                                                  PVWidgets::PVGraphicsViewInteractorBase* interactor)
{
	assert(is_event_supported(type));
	assert(interactor);

	unregister_one(type, interactor);

	_interactor_map[type].push_back(interactor);
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::register_back_all
 *****************************************************************************/


void PVWidgets::PVGraphicsView::register_back_all(PVWidgets::PVGraphicsViewInteractorBase* interactor)
{
	assert(interactor);

	for (QEvent::Type type : _usable_events) {
		_interactor_map[type].push_back(interactor);
	}
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::unregister_one
 *****************************************************************************/
void PVWidgets::PVGraphicsView::unregister_one(QEvent::Type type,
                                               PVWidgets::PVGraphicsViewInteractorBase* interactor)
{
	assert(is_event_supported(type));
	assert(interactor);

	interactor_list_t &ilist = _interactor_map[type];
	interactor_list_t::iterator it = std::find(ilist.begin(), ilist.end(), interactor);

	if (it != ilist.end()) {
		ilist.erase(it);
	}
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::unregister_all
 *****************************************************************************/

void PVWidgets::PVGraphicsView::unregister_all(PVWidgets::PVGraphicsViewInteractorBase* interactor)
{
	assert(interactor);

	for (QEvent::Type type : _usable_events) {
		unregister_one(type, interactor);
	}
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::is_event_supported
 *****************************************************************************/

bool PVWidgets::PVGraphicsView::is_event_supported(QEvent::Type type)
{
	return std::find(_usable_events, _usable_events_end, type) != _usable_events_end;
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::call_interactor
 *****************************************************************************/

bool PVWidgets::PVGraphicsView::call_interactor(QEvent *event)
{
	bool ret = false;
	assert(is_event_supported(event->type()));

	for (PVGraphicsViewInteractorBase* i : _interactor_map[event->type()]) {
		if (i->call(this, event)) {
			ret = true;
			break;
		}
	}

	if (event->isAccepted()) {
		get_viewport()->update();
	}

	return ret;
}

/*****************************************************************************
 * PVWidgets::PVGraphicsView::install_default_scene_interactor
 *****************************************************************************/

void PVWidgets::PVGraphicsView::install_default_scene_interactor()
{
	PVWidgets::PVGraphicsViewInteractorBase *inter =
		declare_interactor<PVWidgets::PVGraphicsViewInteractorScene>();
	register_back_all(inter);

	register_front_one(QEvent::MouseButtonDblClick, inter);
	register_front_one(QEvent::MouseMove, inter);
	register_front_one(QEvent::MouseButtonPress, inter);
	register_front_one(QEvent::MouseButtonRelease, inter);
}
