//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/core/qmetaobject_helper.h>
#include <pvkernel/rush/PVNraw.h>
#include <pvkernel/widgets/PVHelpWidget.h>

#include <squey/PVView.h>

#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVHitCountView.h>
#include <pvparallelview/PVHitGraphData.h>

#include <pvparallelview/PVZoomableDrawingAreaInteractor.h>
#include <pvparallelview/PVZoomableDrawingAreaInteractorMajorY.h>
#include <pvparallelview/PVZoomableDrawingAreaConstraintsMajorY.h>

#include <pvparallelview/PVHitCountViewSelectionRectangle.h>
#include <pvparallelview/PVHitCountViewParamsWidget.h>
#include <pvparallelview/PVHitCountViewInteractor.h>
#include <pvparallelview/PVSelectionRectangleInteractor.h>

#include <QApplication>
#include <QCheckBox>
#include <QGraphicsScene>
#include <QLabel>
#include <QLineEdit>
#include <QPainter>
#include <QResizeEvent>
#include <QScrollBar>
#include <QVBoxLayout>

#include <utility>

#define RENDER_TIMEOUT 75 // in ms

/**
 * @todo how do we want to use that view
 * @todo add sliders
 * @todo make a nice configuration panel
 */

#define print_m(R) __print_mat(#R, R)
#define print_mat(R) __print_mat(#R, R)

template <typename M>
void __print_mat(const char* text, const M& m)
{
	std::cout << text << ": " << std::endl
	          << "  " << m.m11() << " " << m.m12() << " " << m.m13() << std::endl
	          << "  " << m.m21() << " " << m.m22() << " " << m.m23() << std::endl
	          << "  " << m.m31() << " " << m.m32() << " " << m.m33() << std::endl;
}

#define print_r(R) __print_rect(#R, R)
#define print_rect(R) __print_rect(#R, R)

template <typename R>
void __print_rect(const char* text, const R& r)
{
	std::cout << text << ": " << r.x() << " " << r.y() << ", " << r.width() << " " << r.height()
	          << std::endl;
}

#define print_v(R) __print_vect(#R, R)
#define print_vect(R) __print_vect(#R, R)

template <typename R>
void __print_vect(const char* text, const R& r)
{
	std::cout << text << ": " << r.x() << " " << r.y() << std::endl;
}

#define print_s(V) print_scalar(V)
#define print_scalar(V) __print_scalar(#V, V)

template <typename V>
void __print_scalar(const char* text, const V& v)
{
	std::cout << text << ": " << v << std::endl;
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::PVHitCountView
 *****************************************************************************/

PVParallelView::PVHitCountView::PVHitCountView(Squey::PVView& pvview_sp,
                                               create_backend_t create_backend,
                                               const PVCol axis,
                                               QWidget* parent)
    : PVParallelView::PVZoomableDrawingAreaWithAxes(parent)
    , _pvview(pvview_sp)
    , _create_backend(create_backend)
    , _view_deleted(false)
    , _show_bg(true)
    , _auto_x_zoom_sel(false)
    , _do_auto_scale(false)
    , _use_log_color(false)
    , _show_labels(false)
{
	set_gl_viewport();

	if (axis != PVCol()) {
		_backend = _create_backend(axis, this);

		/* computing the highest scene width to setup it... and do the first
		 * run to initialize the manager's buffers :-)
		 */
		get_hit_graph_manager().change_and_process_view(0, 0, .5);
		_max_count = get_hit_graph_manager().get_max_count_all();

		QRectF r(0, 0, _max_count, (size_t)1 << 32);
		set_scene_rect(r);
		get_scene()->setSceneRect(r);
	}

	/* X zoom converter
	 */
	get_x_axis_zoom().set_zoom_converter(&x_zoom_converter());
	get_y_axis_zoom().set_zoom_converter(&y_zoom_converter());

	_sel_rect = new PVHitCountViewSelectionRectangle(this);

	/* interactor
	 */
	_sel_rect_interactor = declare_interactor<PVSelectionRectangleInteractor>(_sel_rect);
	register_front_all(_sel_rect_interactor);

	_my_interactor = declare_interactor<PVZoomableDrawingAreaInteractorMajorY>();
	register_front_all(_my_interactor);
	unregister_one(QEvent::Wheel, _my_interactor);

	_hcv_interactor = declare_interactor<PVHitCountViewInteractor>();
	register_front_one(QEvent::Resize, _hcv_interactor);
	register_front_one(QEvent::KeyPress, _hcv_interactor);
	register_front_one(QEvent::Wheel, _hcv_interactor);

	install_default_scene_interactor();

	// need to move them to front to allow view pan before sel rect move
	register_front_one(QEvent::MouseButtonPress, _my_interactor);
	register_front_one(QEvent::MouseButtonRelease, _my_interactor);
	register_front_one(QEvent::MouseMove, _my_interactor);

	/* constraints
	 */
	set_constraints(new PVParallelView::PVZoomableDrawingAreaConstraintsMajorY());

	/* PVAxisZoom
	 */
	set_x_axis_zoom();
	get_y_axis_zoom().set_range(zoom_min, y_zoom_extra - 1);
	get_y_axis_zoom().set_default_value(zoom_min);
	reset_view();

	/* view configuration
	 */
	set_alignment(Qt::AlignLeft | Qt::AlignTop);
	set_horizontal_scrollbar_policy(Qt::ScrollBarAlwaysOn);
	set_x_legend("Occurrence count");
	auto y_legend = new PVWidgets::PVAxisComboBox(
	    pvview_sp.get_axes_combination(),
	    PVWidgets::PVAxisComboBox::AxesShown::BothOriginalCombinationAxes);
	connect(y_legend, &PVWidgets::PVAxisComboBox::current_axis_changed,
			[this](PVCol axis, PVCombCol){ update_window_title(axis); });
	y_legend->set_current_axis(axis);
	connect(y_legend, &PVWidgets::PVAxisComboBox::current_axis_changed,
	        [this](PVCol axis, PVCombCol) {
		        _backend = _create_backend(axis, this);

		        /* computing the highest scene width to setup it... and do the first
		         * run to initialize the manager's buffers :-)
		         */
		        get_hit_graph_manager().change_and_process_view(0, 0, .5);
		        _max_count = get_hit_graph_manager().get_max_count_all();

		        _sel_rect->set_x_range(0, _max_count);
		        _sel_rect->set_y_range(0, UINT32_MAX);

		        QRectF r(0, 0, _max_count, (uint64_t)1 << 32);
		        set_scene_rect(r);
		        get_scene()->setSceneRect(r);

		        reset_view();

		        recompute_decorations();
		        reconfigure_view();

		        update_all();
		        do_update_all();
	        });
	set_y_legend(y_legend);
	set_decoration_color(Qt::white);
	set_ticks_per_level(8);

	_params_widget = new PVHitCountViewParamsWidget(this);

	_params_widget->setAutoFillBackground(true);
	_params_widget->adjustSize();
	set_params_widget_position();

	_update_all_timer.setInterval(RENDER_TIMEOUT);
	_update_all_timer.setSingleShot(true);
	connect(&_update_all_timer, &QTimer::timeout, this, &PVHitCountView::do_update_all);

	connect(this, &PVHitCountView::zoom_has_changed, this, &PVHitCountView::do_zoom_change);
	connect(this, &PVHitCountView::pan_has_changed, this, &PVHitCountView::do_pan_change);

	connect(get_vertical_scrollbar(), &QScrollBar::valueChanged, this,
	        &PVHitCountView::do_pan_change);

	_help_widget = new PVWidgets::PVHelpWidget(this);
	_help_widget->hide();

	_help_widget->initTextFromFile("hit count view's help");
	_help_widget->addTextFromFile(":help-selection");
	_help_widget->addTextFromFile(":help-layers");
	_help_widget->newColumn();
	_help_widget->addTextFromFile(":help-lines");
	_help_widget->addTextFromFile(":help-application");

	_help_widget->newTable();
	_help_widget->addTextFromFile(":help-mouse-view");
	_help_widget->newColumn();
	_help_widget->addTextFromFile(":help-sel-rect-full");

	_help_widget->newTable();
	_help_widget->addTextFromFile(":help-mouse-hit-count-view");
	_help_widget->newColumn();
	_help_widget->addTextFromFile(":help-shortcuts-hit-count-view");
	_help_widget->finalizeText();

	// Register view for unselected & zombie events toggle
	pvview_sp._toggle_unselected_zombie_visibility.connect(
	    sigc::mem_fun(*this, &PVParallelView::PVHitCountView::toggle_unselected_zombie_visibility));

	_sel_rect->set_default_cursor(Qt::CrossCursor);
	set_viewport_cursor(Qt::CrossCursor);
	set_background_color(color_view_bg);

	_sel_rect->set_x_range(0, _max_count);
	_sel_rect->set_y_range(0, UINT32_MAX);

	_mouse_buttons_default_legend = PVWidgets::PVMouseButtonsLegend("Select", "Pan view", "Zoom (vertical)");
	_mouse_buttons_current_legend = _mouse_buttons_default_legend;
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::~PVHitCountView
 *****************************************************************************/

PVParallelView::PVHitCountView::~PVHitCountView()
{
	if (!_view_deleted) {
		common::get_lib_view(_pvview)->remove_hit_count_view(this);
	}

	delete get_constraints();
	delete _my_interactor;
	delete _hcv_interactor;
	delete _sel_rect_interactor;
	delete _sel_rect;
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::about_to_be_deleted
 *****************************************************************************/

void PVParallelView::PVHitCountView::about_to_be_deleted()
{
	_view_deleted = true;
}

void PVParallelView::PVHitCountView::set_x_axis_zoom()
{
	const int viewport_width = get_margined_viewport_width();

	/* the viewport may have a negative or null size, the X's zoom converter
	 * may generate INF/NaN, so that we had to make sure it is > 0
	 */
	if (viewport_width > 0) {
		const int32_t x_zoom_min = get_x_zoom_min();
		get_x_axis_zoom().set_range(x_zoom_min, x_zoom_converter().scale_to_zoom(viewport_width));
		get_x_axis_zoom().set_default_value(x_zoom_min);
	}
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::update_new_selection_async
 *****************************************************************************/

void PVParallelView::PVHitCountView::update_new_selection_async()
{
	// QMetaObject::invokeMethod(this, &PVHitCountView::update_sel, Qt::QueuedConnection);
	PVCore::invokeMethod(this, &PVHitCountView::update_sel, Qt::QueuedConnection);
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::update_all_async
 *****************************************************************************/

void PVParallelView::PVHitCountView::update_all_async()
{
	// QMetaObject::invokeMethod(this, &PVHitCountView::update_all, Qt::QueuedConnection);
	PVCore::invokeMethod(this, &PVHitCountView::update_all, Qt::QueuedConnection);
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::update_zones
 *****************************************************************************/

bool PVParallelView::PVHitCountView::update_zones()
{
	/* RH: no need to follows the axis_id yet. We need informations to
	 * retrieve the scaled pointer...
	 */
	return true;
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::set_enabled
 *****************************************************************************/

void PVParallelView::PVHitCountView::set_enabled(const bool value)
{
	if (!value) {
		_update_all_timer.stop();
	}

	setDisabled(!value);
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::reset_view
 *****************************************************************************/

void PVParallelView::PVHitCountView::reset_view()
{
	set_zoom_value(PVZoomableDrawingAreaConstraints::X, get_x_zoom_min());
	set_zoom_value(PVZoomableDrawingAreaConstraints::Y, zoom_min);
	_block_zoom_value = get_y_axis_zoom().get_clamped_value();
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::set_x_zoom_level_from_sel
 *****************************************************************************/

void PVParallelView::PVHitCountView::set_x_zoom_level_from_sel()
{
	const uint32_t max_count_sel = get_hit_graph_manager().get_max_count_selected();
	if (max_count_sel > 0) {
		set_zoom_value(PVZoomableDrawingAreaConstraints::X,
		               x_zoom_converter().scale_to_zoom(get_margined_viewport_width() /
		                                                (double)max_count_sel));
	}
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::request_auto_scale
 *****************************************************************************/

void PVParallelView::PVHitCountView::request_auto_scale()
{
	_do_auto_scale = _auto_x_zoom_sel;
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::drawBackground
 *****************************************************************************/

void PVParallelView::PVHitCountView::drawBackground(QPainter* painter, const QRectF& margined_rect)
{
	if (_backend) {
		painter->save();
		const QRect margined_viewport =
		    QRect(-1, -1, get_x_axis_length() + 4, get_y_axis_length() + 2);
		painter->setClipRegion(margined_viewport, Qt::IntersectClip);
		// int img_top = map_margined_from_scene(QPointF(0, _block_base_pos)).y();
		int img_top = map_margined_from_scene(QPointF(0, get_hit_graph_manager().last_y_min())).y();

		int block_view_offset = -img_top;

		int zoom_level = get_y_axis_zoom().get_clamped_value();
		double rel_y_scale = y_zoom_to_scale(zoom_level - _block_zoom_value);

		painter->setPen(Qt::white);

		int x_axis_right = std::min((int)map_margined_from_scene(QPointF(_max_count, 0.)).x(),
		                            get_x_axis_length());

		if (show_bg()) {
			painter->setOpacity(0.25);
			// BENCH_START(hcv_draw_all);
			draw_lines(painter, x_axis_right, block_view_offset, rel_y_scale,
			           get_hit_graph_manager().buffer_all(), 0);

			// painter->setOpacity(0.25);
			// BENCH_START(hcv_draw_selectable);
			draw_lines(painter, x_axis_right, block_view_offset, rel_y_scale,
			           get_hit_graph_manager().buffer_selectable(), 63);
			// BENCH_STOP(hcv_draw_selectable);
			// BENCH_STAT_TIME(hcv_draw_selectable);
		}

		// selected events
		painter->setOpacity(1.0);
		// BENCH_START(hcv_draw_selected);
		draw_lines(painter, x_axis_right, block_view_offset, rel_y_scale,
		           get_hit_graph_manager().buffer_selected(), 255);
		// BENCH_STOP(hcv_draw_selected);
		// BENCH_STAT_TIME(hcv_draw_selected);
		painter->restore();

		draw_decorations(painter, margined_rect);
	}
}

void PVParallelView::PVHitCountView::drawForeground(QPainter* /*painter*/, const QRectF& /*rect*/)
{
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::enterEvent
 *****************************************************************************/

void PVParallelView::PVHitCountView::enterEvent(QEnterEvent* /*event*/)
{
	if (QGuiApplication::keyboardModifiers() == (Qt::ControlModifier | Qt::ShiftModifier)) {
		_mouse_buttons_current_legend.set_left_button_legend("Select (intersection)");
	}
	else if (QGuiApplication::keyboardModifiers() == Qt::ControlModifier) {
		_mouse_buttons_current_legend.set_left_button_legend("Select (substraction)");
		_mouse_buttons_current_legend.set_scrollwheel_legend("Resize (local)");
	}
	else if (QGuiApplication::keyboardModifiers() == Qt::ShiftModifier) {
		_mouse_buttons_current_legend.set_left_button_legend("Select (union)");
	}
	Q_EMIT set_status_bar_mouse_legend(_mouse_buttons_current_legend);
	setFocus(Qt::MouseFocusReason);
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::leaveEvent
 *****************************************************************************/

void PVParallelView::PVHitCountView::leaveEvent(QEvent*)
{
	Q_EMIT clear_status_bar_mouse_legend();
	_mouse_buttons_current_legend = _mouse_buttons_default_legend;
	clearFocus();
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::keyPressEvent
 *****************************************************************************/

void PVParallelView::PVHitCountView::keyPressEvent(QKeyEvent* event)
{
	if (event->modifiers() == (Qt::ControlModifier | Qt::ShiftModifier)) {
		_mouse_buttons_current_legend.set_left_button_legend("Select (intersection)");
		_mouse_buttons_current_legend.set_scrollwheel_legend("Zoom");
		Q_EMIT set_status_bar_mouse_legend(_mouse_buttons_current_legend);
	}
	if (event->modifiers() == Qt::ControlModifier) {
		_mouse_buttons_current_legend.set_left_button_legend("Select (substraction)");
		_mouse_buttons_current_legend.set_scrollwheel_legend("Zoom");
		Q_EMIT set_status_bar_mouse_legend(_mouse_buttons_current_legend);
	}
	else if (event->modifiers() == Qt::ShiftModifier) {
		_mouse_buttons_current_legend.set_left_button_legend("Select (union)");
		_mouse_buttons_current_legend.set_scrollwheel_legend("Zoom (horizontal)");
		Q_EMIT set_status_bar_mouse_legend(_mouse_buttons_current_legend);
	}

	PVZoomableDrawingAreaWithAxes::keyPressEvent(event);
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::keyReleaseEvent
 *****************************************************************************/

void PVParallelView::PVHitCountView::keyReleaseEvent(QKeyEvent* event)
{
	if (event->key() == Qt::Key_Control and event->modifiers() == Qt::ShiftModifier) {
		_mouse_buttons_current_legend.set_left_button_legend("Select (union)");
		_mouse_buttons_current_legend.set_scrollwheel_legend("Zoom (horizontal)");
		Q_EMIT set_status_bar_mouse_legend(_mouse_buttons_current_legend);
	}
	else if (event->key() == Qt::Key_Shift and event->modifiers() == Qt::ControlModifier) {
		_mouse_buttons_current_legend.set_left_button_legend("Select (substraction)");
		_mouse_buttons_current_legend.set_scrollwheel_legend("Zoom");
		Q_EMIT set_status_bar_mouse_legend(_mouse_buttons_current_legend);
	}
	else if (event->key() == Qt::Key_Control or event->key() == Qt::Key_Shift) {
		_mouse_buttons_current_legend.set_left_button_legend("Select");
		_mouse_buttons_current_legend.set_scrollwheel_legend("Zoom (vertical)");
		Q_EMIT set_status_bar_mouse_legend(_mouse_buttons_current_legend);
	}

	PVZoomableDrawingAreaWithAxes::keyReleaseEvent(event);
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::draw_lines
 *****************************************************************************/

void PVParallelView::PVHitCountView::draw_lines(QPainter* painter,
                                                const int x_max,
                                                const int block_view_offset,
                                                const double rel_y_scale,
                                                const uint32_t* buffer,
                                                const int hsv_value)
{
	const int y_axis_length = get_y_axis_length();
	const size_t buffer_size = get_hit_graph_manager().size_int();

	uint32_t min_value = UINT32_MAX;
	uint32_t max_value = 0;

	for (uint32_t idx = 0; idx < buffer_size; ++idx) {
		const uint32_t count = buffer[idx];
		if (count < min_value) {
			min_value = count;
		}
		if (count > max_value) {
			max_value = count;
		}
	}

	const double range_value = max_value - min_value;
	const double log_range_value = log(range_value);

	for (uint32_t idx = 0; idx < buffer_size; ++idx) {
		const uint32_t count = buffer[idx];
		if (count == 0) {
			continue;
		}

		int y_val = (rel_y_scale * idx) - block_view_offset;

		if ((y_val < 0) || (y_val >= y_axis_length)) {
			continue;
		}

		int vx = map_margined_from_scene(QPointF(count, 0.)).x();

		if (vx < 0) {
			continue;
		}

		vx = std::min(x_max, vx);

		double ratio = count - min_value;

		if (use_log_color()) {
			ratio = log(ratio) / log_range_value;
		} else {
			ratio /= range_value;
		}

		// using fillRect is faster than drawLine... 10 times faster
		// for the color-ramp, it's: ratio * (RED - GREEN) + GREEN
		painter->fillRect(0, y_val, vx, 1,
		                  QColor::fromHsv(ratio * (0 - 240) + 240, 255, hsv_value));
	}
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::do_zoom_change
 *****************************************************************************/

void PVParallelView::PVHitCountView::do_zoom_change(int axes)
{
	if (not _backend) {
		return;
	}
	if (axes & PVZoomableDrawingAreaConstraints::Y) {
		if (_do_auto_scale) {
			get_horizontal_scrollbar()->setValue(0);
		}
	}
	_sel_rect->set_handles_scale(1. / get_transform().m11(), 1. / get_transform().m22());

	get_y_labels_cache().invalidate();
	_update_all_timer.start();
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::do_pan_change
 *****************************************************************************/

void PVParallelView::PVHitCountView::do_pan_change()
{
	if (not _backend) {
		return;
	}
	_update_all_timer.start();
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::do_update_all
 *****************************************************************************/

void PVParallelView::PVHitCountView::do_update_all()
{
	if (not _backend) {
		return;
	}

	int zoom_value, zoom_level;
	double alpha;

	zoom_value = get_y_axis_zoom().get_clamped_value();

	if (zoom_value < 0) {
		int rel_zoom = zoom_value - zoom_min;
		zoom_level = rel_zoom / zoom_steps;
		alpha = 0.5 * _y_zoom_converter.zoom_to_scale_decimal(rel_zoom);
	} else {
		zoom_value = 0;
		zoom_level = digital_zoom_level;
		alpha = 1.0;
	}

	uint32_t y_min = map_margined_to_scene(0, 0).y();
	uint64_t block_size = 1L << (32 - zoom_level);
	uint32_t block_y_min = (uint64_t)y_min & ~(block_size - 1);

	get_hit_graph_manager().change_and_process_view(block_y_min, zoom_level, alpha);

	if (_do_auto_scale) {
		set_x_zoom_level_from_sel();

		const ViewportAnchor old_anchor = get_transformation_anchor();
		set_transformation_anchor(PVGraphicsView::NoAnchor);
		reconfigure_view();

		_sel_rect->set_handles_scale(1. / get_transform().m11(), 1. / get_transform().m22());

		set_transformation_anchor(old_anchor);

		_do_auto_scale = false;
	}

	_block_zoom_value = zoom_value;

	get_viewport()->update();
}

/******************************************************************************
 * PVParallelView::PVHitCountView::toggle_unselected_zombie_visibility
 *****************************************************************************/

void PVParallelView::PVHitCountView::toggle_unselected_zombie_visibility()
{
	_show_bg = _pvview.are_view_unselected_zombie_visible();

	get_viewport()->update();
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::update_all
 *****************************************************************************/

void PVParallelView::PVHitCountView::update_all()
{
	if (not _backend) {
		return;
	}
	get_hit_graph_manager().process_all_buffers();
	get_viewport()->update();
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::update_sel
 *****************************************************************************/

void PVParallelView::PVHitCountView::update_sel()
{
	if (not _backend) {
		return;
	}
	get_hit_graph_manager().process_buffer_selected();
	if (_auto_x_zoom_sel) {
		set_x_zoom_level_from_sel();
		set_transformation_anchor(PVGraphicsView::NoAnchor);
		reconfigure_view();

		_sel_rect->set_handles_scale(1. / get_transform().m11(), 1. / get_transform().m22());

		set_transformation_anchor(PVGraphicsView::AnchorUnderMouse);
		get_horizontal_scrollbar()->setValue(0);
	}

	get_viewport()->update();
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::toggle_auto_x_zoom_sel
 *****************************************************************************/

void PVParallelView::PVHitCountView::toggle_auto_x_zoom_sel()
{
	_auto_x_zoom_sel = !_auto_x_zoom_sel;
	params_widget()->update_widgets();

	request_auto_scale();

	if (_auto_x_zoom_sel) {
		do_update_all();
	}
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::toggle_log_color
 *****************************************************************************/

void PVParallelView::PVHitCountView::toggle_log_color()
{
	_use_log_color = !_use_log_color;
	params_widget()->update_widgets();

	get_viewport()->update();
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::toggle_show_labels
 *****************************************************************************/

void PVParallelView::PVHitCountView::toggle_show_labels()
{
	_show_labels = !_show_labels;
	params_widget()->update_widgets();

	if (_show_labels) {
		PVCore::PVProgressBox::progress(
		    [&](PVCore::PVProgressBox& pbox) {
			    pbox.set_enable_cancel(false);
			    pbox.set_extended_status("Computing Y-axis labels index");
			    get_y_labels_cache().initialize();
		    },
		    "Initializing labels index...", this);
	}

	recompute_decorations();
	reconfigure_view();
	get_viewport()->update();
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::set_params_widget_position
 *****************************************************************************/

void PVParallelView::PVHitCountView::set_params_widget_position()
{
	QPoint pos(get_viewport()->width() - frame_offsets.right(), frame_offsets.top());

	pos -= QPoint(_params_widget->width(), 0);
	_params_widget->move(pos);
	_params_widget->raise();
}

QString PVParallelView::PVHitCountView::get_x_value_at(const qint64 value)
{
	if (_show_labels) {
		// Number of Occurrence
		return get_elided_text(QString::number(value));
	} else {
		return {};
	}
}

QString PVParallelView::PVHitCountView::get_y_value_at(const qint64 value)
{
	if (_show_labels) {
		return get_elided_text(get_y_labels_cache().get(value));
	} else {
		return {};
	}
}

void PVParallelView::PVHitCountView::update_window_title(PVCol axis)
{
	setWindowTitle(QString("%1 (%2)").arg(
		QObject::tr("Hit count"),
		_pvview.get_nraw_axis_name(axis)));
}
