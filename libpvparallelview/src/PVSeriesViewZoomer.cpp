#include <pvparallelview/PVSeriesViewZoomer.h>

#include <pvparallelview/PVSeriesView.h>
#include <inendi/PVRangeSubSampler.h>

#include <QMouseEvent>
#include <QToolTip>
#include <QPainter>

#include <QTimer>
#include <QDebug>

namespace PVParallelView
{

PVViewZoomer::PVViewZoomer(QWidget* parent) : QWidget(parent)
{
	_zoom_stack.push_back(Zoom{0., 1., 0., 1.});
}

void PVViewZoomer::zoom_in(QRectF zoom_in_rect)
{
	qDebug() << "zoom_in" << zoom_in_rect;
	_zoom_stack.resize(_current_zoom_index + 1);
	_zoom_stack.push_back(rect_to_zoom(zoom_in_rect));
	++_current_zoom_index;
	update_zoom();
}

void PVViewZoomer::zoom_in(QPointF center, bool rectangular, zoom_f zoom_factor)
{
	qDebug() << "zoom_in" << center;
	assert(zoom_factor < 1 && zoom_factor > 0);
	zoom_in(QRectF(center.x() - center.x() * zoom_factor,
	               center.y() - center.y() * (rectangular ? zoom_factor : 1),
	               zoom_factor * size().width(), (rectangular ? zoom_factor : 1) * size().height()));
}

void PVViewZoomer::zoom_out()
{
	qDebug() << "zoom_out";
	if (_current_zoom_index == 0) {
		return;
	}
	--_current_zoom_index;
	update_zoom();
}

void PVViewZoomer::zoom_out(QPointF center, Zoom old_zoom, Zoom target_zoom)
{
	qDebug() << "zoom_out" << center;

	struct PointF {
		zoom_f _x;
		zoom_f _y;
		zoom_f x() const { return _x; }
		zoom_f y() const { return _y; }
		zoom_f& rx() { return _x; }
		zoom_f& ry() { return _y; }
	};

	PointF center_ratio{zoom_f(center.x()) / zoom_f(size().width()),
	                    1. - zoom_f(center.y()) / zoom_f(size().height())};
	PointF fixed_center{old_zoom.minX + center_ratio.x() * old_zoom.width(),
	                    old_zoom.minY + center_ratio.y() * old_zoom.height()};
	PointF target_top_left{fixed_center.x() - center_ratio.x() * target_zoom.width(),
	                       fixed_center.y() - center_ratio.y() * target_zoom.height()};

	target_top_left.rx() =
	    std::clamp(target_top_left.x(), zoom_f(0), zoom_f(1) - target_zoom.width());
	target_top_left.ry() =
	    std::clamp(target_top_left.y(), zoom_f(0), zoom_f(1) - target_zoom.height());

	_zoom_stack[_current_zoom_index] =
	    Zoom{target_top_left.x(), target_top_left.x() + target_zoom.width(), target_top_left.y(),
	         target_top_left.y() + target_zoom.height()};

	update_zoom();
}

void PVViewZoomer::zoom_out(QPointF center)
{
	if (_current_zoom_index == 0) {
		return;
	}
	Zoom old_zoom = _zoom_stack[_current_zoom_index];
	--_current_zoom_index;
	Zoom target_zoom = _zoom_stack[_current_zoom_index];
	zoom_out(center, old_zoom, target_zoom);
}

void PVViewZoomer::zoom_out(QPointF center, bool rectangular, zoom_f zoom_factor)
{
	if (_current_zoom_index == 0) {
		return;
	}
	Zoom old_zoom = _zoom_stack[_current_zoom_index];
	Zoom target_zoom = old_zoom;
	target_zoom.minX = 0;
	target_zoom.maxX = std::min(old_zoom.width() / zoom_factor, zoom_f(1));
	if (rectangular) {
		target_zoom.minY = 0;
		target_zoom.maxY = std::min(old_zoom.height() / zoom_factor, zoom_f(1));
	}
	_zoom_stack.resize(2);
	_current_zoom_index = 1;
	zoom_out(center, old_zoom, target_zoom);
}

void PVViewZoomer::reset_zoom()
{
	_zoom_stack.clear();
	_zoom_stack.push_back(Zoom{0., 1., 0., 1.});
	_current_zoom_index = 0;
	update_zoom();
}

void PVViewZoomer::reset_and_zoom_in(Zoom zoom)
{
	_zoom_stack.clear();
	_zoom_stack.push_back(Zoom{0., 1., 0., 1.});
	_zoom_stack.push_back(zoom);
	_current_zoom_index = 1;
	update_zoom();
}

void PVViewZoomer::move_zoom_by(QPoint offset)
{
	if (_current_zoom_index == 0) {
		return;
	}
	_zoom_stack.resize(_current_zoom_index + 1);
	Zoom& zoom = _zoom_stack.back();
	{
		zoom_f offsetX = offset.x() * zoom.width() / zoom_f(size().width());
		offsetX = std::clamp(offsetX, -(1. - zoom.maxX), zoom.minX);
		zoom.minX -= offsetX;
		zoom.maxX -= offsetX;
	}
	{
		zoom_f offsetY = offset.y() * zoom.height() / zoom_f(size().height());
		offsetY = std::clamp(offsetY, -zoom.minY, 1. - zoom.maxY);
		zoom.minY += offsetY;
		zoom.maxY += offsetY;
	}
	update_zoom();
}

QRect PVViewZoomer::normalized_zoom_rect(QRect zoom_rect, bool rectangular) const
{
	auto zr = zoom_rect.normalized();
	if (not rectangular) {
		zr.setY(0);
		zr.setHeight(size().height());
	}
	return zr;
}

auto PVViewZoomer::rect_to_zoom(QRectF const& rect) const -> Zoom
{
	Zoom const& current_zoom = _zoom_stack[_current_zoom_index];
	return Zoom{current_zoom.minX + current_zoom.width() * (rect.x() / zoom_f(size().width())),
	            current_zoom.minX +
	                current_zoom.width() * ((rect.x() + rect.width()) / zoom_f(size().width())),
	            current_zoom.minY + current_zoom.height() *
	                                    (1. - (rect.y() + rect.height()) / zoom_f(size().height())),
	            current_zoom.minY +
	                current_zoom.height() * (1. - rect.y() / zoom_f(size().height()))};
}

auto PVViewZoomer::clamp_zoom(Zoom zoom) -> Zoom
{
	zoom.minX = std::clamp(zoom.minX, zoom_f(0), zoom_f(1));
	zoom.maxX = std::clamp(zoom.maxX, zoom_f(0), zoom_f(1));
	zoom.minY = std::clamp(zoom.minY, zoom_f(0), zoom_f(1));
	zoom.maxY = std::clamp(zoom.maxY, zoom_f(0), zoom_f(1));
	return zoom;
}

void PVViewZoomer::update_zoom()
{
	update_zoom(current_zoom());
	zoom_updated(current_zoom());

	// qDebug() << "Zoom stack (current:" << _current_zoom_index << "):";
	// for (auto& zoom : _zoom_stack) {
	// 	qDebug() << "\t" << std::to_string(zoom.minX).c_str() << std::to_string(zoom.maxX).c_str()
	// 	         << std::to_string(zoom.minY).c_str() << std::to_string(zoom.maxY).c_str();
	// }
}

struct PVSeriesViewZoomerRectangleFragment : public QWidget {
	PVSeriesViewZoomerRectangleFragment(QWidget* parent = nullptr,
	                                    QColor color = QColor(255, 0, 0, 255))
	    : QWidget(parent), color(color)
	{
	}

	void paintEvent(QPaintEvent*) override
	{
		QPainter painter(this);
		painter.fillRect(rect(), color);
	}

	QColor color;
};

PVSeriesViewZoomer::PVSeriesViewZoomer(PVSeriesView* child,
                                       Inendi::PVRangeSubSampler& sampler,
                                       QWidget* parent)
    : PVViewZoomer(parent), _series_view(child), _rss(sampler), _animation_timer(new QTimer(this))
{
	child->setParent(this);
	child->setAttribute(Qt::WA_TransparentForMouseEvents);
	for (auto& fragment : _selector_fragments) {
		fragment = new PVSeriesViewZoomerRectangleFragment(
		    child, _selector_colors[size_t(_selector_mode)]);
		fragment->hide();
	}
	for (auto& chronotip : _chronotips) {
		chronotip = new QLabel(child);
		chronotip->setSizePolicy(QSizePolicy::Policy::Expanding, QSizePolicy::Policy::Minimum);
		chronotip->setAutoFillBackground(true);
		chronotip->hide();
	}
	_chronotips[0]->setAlignment(Qt::AlignRight);
	_chronotips[1]->setAlignment(Qt::AlignLeft);
	setFocusPolicy(Qt::StrongFocus);
	setMouseTracking(true);
	setCursor(Qt::BlankCursor);
}

void PVSeriesViewZoomer::change_selector_mode(SelectorMode const mode)
{
	const auto previous_mode = _selector_mode;
	if (mode == previous_mode) {
		return;
	}
	if (mode == SelectorMode::CrossHairs) {
		_selector_mode = SelectorMode::CrossHairs;
		_selector_rect = QRect(QPoint(_selector_rect.left() + _selector_rect.width(),
		                              _selector_rect.top() + _selector_rect.height()),
		                       QSize(0, 0));
		update_selector_and_chronotips();
	}
	if (previous_mode == SelectorMode::CrossHairs) {
		_selector_mode = mode;
		_selector_rect = QRect(_selector_rect.topLeft(), QSize(0, 0));
		update_selector_and_chronotips();
	} else {
		_selector_mode = mode;
		update_selector_and_chronotips();
	}
	if (_selector_mode != previous_mode) {
		selector_mode_changed(previous_mode, _selector_mode);
	}
}

void PVSeriesViewZoomer::mousePressEvent(QMouseEvent* event)
{
	if (event->button() == Qt::LeftButton) {
		_left_button_down = true;
		if (_selector_mode == SelectorMode::CrossHairs) {
			if (event->modifiers() & Qt::ShiftModifier and not _selecting_mode_disabled) {
				change_selector_mode(SelectorMode::Selecting);
			} else {
				change_selector_mode(SelectorMode::Zooming);
			}
		} else {
			_selector_rect = QRect(event->pos(), QSize(0, 0));
			update_selector_and_chronotips();
		}
	}

	if (event->button() == Qt::RightButton) {
		_moving = true;
		_move_start = event->pos();
	} else if (event->button() == Qt::MidButton) {
		using namespace std::chrono;
		_animation_timer->callOnTimeout([this] { move_zoom_by({-1, 0}); });
		_animation_timer->start(36ms);
	}
}

void PVSeriesViewZoomer::mouseReleaseEvent(QMouseEvent* event)
{
	if (event->button() == Qt::LeftButton) {
		_left_button_down = false;
		if (_selector_mode == SelectorMode::Zooming) {
			if (event->pos() == _selector_rect.topLeft()) {
				zoom_out(event->pos());
			} else {
				zoom_in(normalized_zoom_rect(_selector_rect & rect(),
				                             event->modifiers() & Qt::ControlModifier));
			}
			change_selector_mode(SelectorMode::CrossHairs);
		} else if (_selector_mode == SelectorMode::Selecting) {
			selection_commit(clamp_zoom(rect_to_zoom(normalized_zoom_rect(_selector_rect, false))));
			change_selector_mode(SelectorMode::CrossHairs);
		} else if (_selector_mode == SelectorMode::Hunting) {
			hunt_commit(_selector_rect.isNull() ? cross_hairs_rect(_selector_rect.topLeft())
			                                    : normalized_zoom_rect(_selector_rect, true),
			            not(event->modifiers() & Qt::ControlModifier));
			change_selector_mode(SelectorMode::CrossHairs);
		}
	}

	if (_moving && event->button() == Qt::RightButton) {
		_moving = false;
		move_zoom_by(event->pos() - _move_start);
	} else if (event->button() == Qt::MidButton) {
		_animation_timer->stop();
	}
}

void PVSeriesViewZoomer::mouseMoveEvent(QMouseEvent* event)
{
	if (not(event->buttons() & Qt::LeftButton)) {
		_selector_rect.setTopLeft(event->pos());
	}
	_selector_rect.setSize(
	    QSize(std::clamp(event->pos().x(), 0, size().width()) - _selector_rect.left(),
	          std::clamp(event->pos().y(), 0, size().height()) - _selector_rect.top()));
	update_selector_and_chronotips();
	if (_selector_mode == SelectorMode::Hunting) {
		if (event->buttons() & Qt::LeftButton) {
			cursor_moved(normalized_zoom_rect(_selector_rect, true));
		} else {
			cursor_moved(cross_hairs_rect(_selector_rect.topLeft()));
		}
	}
	if (_moving) {
		move_zoom_by(event->pos() - _move_start);
		_move_start = event->pos();
		update_chronotips(event->pos());
	}
}

void PVSeriesViewZoomer::keyPressEvent(QKeyEvent* event)
{
	if (event->key() == Qt::Key_Control) {
		_control_modifier = true;
		update_selector_and_chronotips();
		return;
	} else if (event->key() == Qt::Key_X) {
		if (_selector_mode == SelectorMode::CrossHairs) {
			change_selector_mode(SelectorMode::Hunting);
			cursor_moved(cross_hairs_rect(_selector_rect.topLeft()));
			return;
		} else if (_selector_mode == SelectorMode::Hunting) {
			change_selector_mode(SelectorMode::CrossHairs);
			return;
		}
	} else if (event->key() == Qt::Key_S) {
		if (_selector_mode == SelectorMode::CrossHairs and not _selecting_mode_disabled) {
			change_selector_mode(SelectorMode::Selecting);
			cursor_moved(cross_hairs_rect(_selector_rect.topLeft()));
			return;
		} else if (_selector_mode == SelectorMode::Selecting) {
			change_selector_mode(SelectorMode::CrossHairs);
			return;
		}
	} else if (event->key() == Qt::Key_Escape) {
		change_selector_mode(SelectorMode::CrossHairs);
		return;
	}
	QWidget::keyPressEvent(event);
}

void PVSeriesViewZoomer::keyReleaseEvent(QKeyEvent* event)
{
	if (event->key() == Qt::Key_Control) {
		_control_modifier = false;
		update_selector_and_chronotips();
		return;
	}
	QWidget::keyReleaseEvent(event);
}

void PVSeriesViewZoomer::enterEvent(QEvent*)
{
	activateWindow();
	setFocus(Qt::MouseFocusReason);
}

void PVSeriesViewZoomer::leaveEvent(QEvent*)
{
	clearFocus();
	if (not _left_button_down) {
		hide_fragments(_selector_fragments);
		hide_fragments(_chronotips);
	}
}

void PVSeriesViewZoomer::wheelEvent(QWheelEvent* event)
{
	if (event->angleDelta().y() > 0) {
		zoom_in(event->position(), event->modifiers() & Qt::ControlModifier, _centered_zoom_factor);
	} else if (event->angleDelta().y() < 0) {
		zoom_out(event->position(), event->modifiers() & Qt::ControlModifier, _centered_zoom_factor);
	}
}

void PVSeriesViewZoomer::resizeEvent(QResizeEvent*)
{
	_resizing_timer.stop();
	_resizing_timer.start(200, this);
	_series_view->resize(size());
}

void PVSeriesViewZoomer::timerEvent(QTimerEvent* event)
{
	if (event->timerId() == _resizing_timer.timerId()) {
		if (_rss.samples_count() != size_t(size().width())) {
			_rss.set_sampling_count(size().width());
			update_zoom(current_zoom());
		} else {
			_series_view->refresh();
		}
		_resizing_timer.stop();
	}
}

void PVSeriesViewZoomer::update_selector_and_chronotips()
{
	if (_selector_mode == SelectorMode::CrossHairs or _selector_rect.isNull()) {
		update_cross_hairs_geometry(_selector_rect.topLeft());
	} else if (_selector_mode == SelectorMode::Zooming) {
		update_selector_geometry(_control_modifier);
	} else if (_selector_mode == SelectorMode::Selecting) {
		update_selector_geometry(false);
	} else if (_selector_mode == SelectorMode::Hunting) {
		update_selector_geometry(true);
	}
	if (_selector_rect.isNull() and rect().contains(_selector_rect.topLeft())) {
		update_chronotips(_selector_rect.topLeft());
	} else {
		update_chronotips(_selector_rect & rect());
	}
}

void PVSeriesViewZoomer::update_selector_geometry(bool rectangular)
{
	auto& fragments = _selector_fragments;
	for (auto& fragment : fragments) {
		static_cast<PVSeriesViewZoomerRectangleFragment*>(fragment)->color =
		    _selector_colors[size_t(_selector_mode)];
	}
	auto nr = normalized_zoom_rect(_selector_rect, rectangular);
	fragments[1]->setGeometry(nr.x(), nr.y(), 1, nr.height());
	fragments[3]->setGeometry(nr.x() + nr.width(), nr.y(), 1, nr.height());
	fragments[1]->show();
	fragments[3]->show();
	if (rectangular) {
		fragments[0]->setGeometry(nr.x(), nr.y(), nr.width(), 1);
		fragments[2]->setGeometry(nr.x(), nr.y() + nr.height(), nr.width(), 1);
		fragments[0]->show();
		fragments[2]->show();
	} else {
		fragments[0]->hide();
		fragments[2]->hide();
	}
}

void PVSeriesViewZoomer::update_cross_hairs_geometry(QPoint pos)
{
	auto& fragments = _selector_fragments;
	for (auto& fragment : fragments) {
		static_cast<PVSeriesViewZoomerRectangleFragment*>(fragment)->color =
		    _selector_colors[size_t(_selector_mode)];
	}
	fragments[0]->setGeometry(0, pos.y(), pos.x() - _cross_hairs_radius, 1);
	fragments[1]->setGeometry(pos.x(), 0, 1, pos.y() - _cross_hairs_radius);
	fragments[2]->setGeometry(pos.x() + _cross_hairs_radius, pos.y(),
	                          width() - pos.x() - _cross_hairs_radius, 1);
	fragments[3]->setGeometry(pos.x(), pos.y() + _cross_hairs_radius, 1,
	                          height() - pos.y() - _cross_hairs_radius);
	show_fragments(fragments);
}

void PVSeriesViewZoomer::update_chronotip_geometry(size_t chrono_index, QPoint pos)
{
	bool chronotips_top = (pos.y() < height() / 2)
	                          ? pos.y() > 3 * _chronotips[0]->height()
	                          : pos.y() > height() - 3 * _chronotips[0]->height();
	bool force_chronotips_right = pos.x() < _chronotips[0]->width();
	bool force_chronotips_left = pos.x() > width() - _chronotips[1]->width();
	if (chrono_index == 0) {
		int chronotips_y_0 =
		    chronotips_top ? 0
		                   : force_chronotips_left or force_chronotips_right
		                         ? height() - _chronotips[0]->height() - _chronotips[1]->height()
		                         : height() - _chronotips[0]->height();
		_chronotips[0]->move(force_chronotips_right ? pos.x() + 1
		                                            : pos.x() - _chronotips[0]->width(),
		                     chronotips_y_0);
	} else if (chrono_index == 1) {
		int chronotips_y_1 =
		    chronotips_top
		        ? force_chronotips_left or force_chronotips_right ? _chronotips[0]->height() : 0
		        : height() - _chronotips[1]->height();
		_chronotips[1]->move(force_chronotips_left ? pos.x() - _chronotips[1]->width()
		                                           : pos.x() + 1,
		                     chronotips_y_1);
	} else if (chrono_index == 2) {
		bool percenttips_left = pos.x() > 2 * _chronotips[2]->width();
		_chronotips[2]->move(percenttips_left ? 0 : width() - _chronotips[2]->width(), pos.y() + 1);
	} else if (chrono_index == 3) {
		bool percenttips_left = pos.x() > 2 * _chronotips[3]->width();
		_chronotips[3]->move(percenttips_left ? 0 : width() - _chronotips[3]->width(),
		                     pos.y() - _chronotips[3]->height());
	}
}

template <class T>
void PVSeriesViewZoomer::show_fragments(T const& fragments) const
{
	for (auto& fragment : fragments) {
		fragment->show();
		fragment->update();
	}
}

template <class T>
void PVSeriesViewZoomer::hide_fragments(T const& fragments) const
{
	for (auto& fragment : fragments) {
		fragment->hide();
	}
}

void PVSeriesViewZoomer::update_zoom(Zoom zoom)
{
	_rss.subsample(zoom.minX, zoom.maxX, zoom.minY, zoom.maxY);
}

void PVSeriesViewZoomer::update_chronotips(QPoint point)
{
	update_chronotips(QRect(point.x(), point.y(), 1, 1));
}

void PVSeriesViewZoomer::update_chronotips(QRect rect_in)
{
	auto rect = normalized_zoom_rect(rect_in, true);
	Zoom coveredZoom = clamp_zoom(rect_to_zoom(rect));
	pvcop::db::array subrange = _rss.ratio_to_minmax(coveredZoom.minX, coveredZoom.maxX);
	_chronotips[0]->setText((subrange.at(0) + ">").c_str());
	_chronotips[1]->setText(("<" + subrange.at(1)).c_str());
	_chronotips[2]->setText((std::to_string(coveredZoom.minY) + "%").c_str());
	_chronotips[3]->setText((std::to_string(coveredZoom.maxY) + "%").c_str());
	for (auto* chronotip : _chronotips) {
		chronotip->adjustSize();
	}
	update_chronotip_geometry(0, rect.topLeft());
	update_chronotip_geometry(1, rect.bottomRight());
	update_chronotip_geometry(2, rect.bottomRight());
	update_chronotip_geometry(3, rect.topLeft());
	show_fragments(_chronotips);
}

QRect PVSeriesViewZoomer::cross_hairs_rect(QPoint pos) const
{
	return QRect(pos.x() - _cross_hairs_radius, pos.y() - _cross_hairs_radius,
	             2 * _cross_hairs_radius + 1, 2 * _cross_hairs_radius + 1)
	    .intersected(rect());
}
} // namespace PVParallelView
