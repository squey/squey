
#include <pvkernel/widgets/PVConfigPopupWidget.h>
#include <pvkernel/widgets/PVTextPopupWidget.h>

#include <picviz/PVView.h>

#include <pvhive/PVHive.h>

#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVHitCountView.h>
#include <pvparallelview/PVHitGraphData.h>

#include <pvparallelview/PVZoomableDrawingAreaInteractor.h>
#include <pvparallelview/PVZoomableDrawingAreaInteractorMajorY.h>
#include <pvparallelview/PVZoomableDrawingAreaConstraintsMajorY.h>
#include <pvparallelview/PVSelectionRectangleInteractor.h>
#include <pvparallelview/PVSelectionSquare.h>
#include <pvparallelview/PVSelectionRectangleInteractor.h>

#include <QCheckBox>
#include <QGraphicsScene>
#include <QLabel>
#include <QLineEdit>
#include <QPainter>
#include <QResizeEvent>
#include <QScrollBar64>
#include <QVBoxLayout>

#define RENDER_TIMEOUT 75 // in ms

/**
 * @todo how do we want to use that view
 * @todo add sliders
 * @todo make a nice configuration panel
 */

#define print_m(R) __print_mat(#R, R)
#define print_mat(R) __print_mat(#R, R)

template <typename M>
void __print_mat(const char *text, const M &m)
{
	std::cout << text << ": " << std::endl
	          << "  " << m.m11() << " " << m.m12() << " " << m.m13() << std::endl
	          << "  " << m.m21() << " " << m.m22() << " " << m.m23() << std::endl
	          << "  " << m.m31() << " " << m.m32() << " " << m.m33() << std::endl;
}

#define print_r(R) __print_rect(#R, R)
#define print_rect(R) __print_rect(#R, R)

template <typename R>
void __print_rect(const char *text, const R &r)
{
	std::cout << text << ": "
	          << r.x() << " " << r.y() << ", "
	          << r.width() << " " << r.height()
	          << std::endl;
}

#define print_v(R) __print_vect(#R, R)
#define print_vect(R) __print_vect(#R, R)

template <typename R>
void __print_vect(const char *text, const R &r)
{
	std::cout << text << ": "
	          << r.x() << " " << r.y()
	          << std::endl;
}

#define print_s(V) print_scalar(V)
#define print_scalar(V) __print_scalar(#V, V)

template <typename V>
void __print_scalar(const char *text, const V &v)
{
	std::cout << text << ": "
	          << v
	          << std::endl;
}

namespace PVParallelView
{

class PVHitCountViewParamsWidget: public PVWidgets::PVConfigPopupWidget
{
public:
	PVHitCountViewParamsWidget(PVHitCountView* parent) :
		PVWidgets::PVConfigPopupWidget(parent)
	{
		setWindowTitle(tr("Hit count view - options"));

		_cb_autofit = new QCheckBox(tr("Auto-fit selection on the occurence axis"));
		_cb_use_log_color = new QCheckBox(tr("Use logarithmic colormap"));

		QVBoxLayout* layout = new QVBoxLayout();
		layout->addWidget(_cb_autofit);
		layout->addWidget(_cb_use_log_color);
		setContentLayout(layout);

		connect(_cb_autofit, SIGNAL(toggled(bool)),
		        parent_hcv(), SLOT(toggle_auto_x_zoom_sel()));
		connect(_cb_use_log_color,  SIGNAL(toggled(bool)),
		        parent_hcv(), SLOT(toggle_log_color()));
	}

public:
	void update_widgets()
	{
		_cb_autofit->blockSignals(true);
		_cb_use_log_color->blockSignals(true);

		_cb_autofit->setChecked(parent_hcv()->auto_x_zoom_sel());
		_cb_use_log_color->setChecked(parent_hcv()->use_log_color());

		_cb_autofit->blockSignals(false);
		_cb_use_log_color->blockSignals(false);
	}

private:
	PVHitCountView* parent_hcv()
	{
		assert(qobject_cast<PVHitCountView*>(parentWidget()));
		return static_cast<PVHitCountView*>(parentWidget());
	}

private:
	QCheckBox* _cb_autofit;
	QCheckBox* _cb_use_log_color;
};

class PVHitCountViewInteractor : public PVZoomableDrawingAreaInteractor
{
public:
	PVHitCountViewInteractor(PVWidgets::PVGraphicsView* parent = nullptr) :
		PVZoomableDrawingAreaInteractor(parent)
	{}

	bool resizeEvent(PVZoomableDrawingArea* zda, QResizeEvent*) override
	{
		PVHitCountView *hcv = get_hit_count_view(zda);
		hcv->set_x_axis_zoom();

		hcv->reconfigure_view();
		if (hcv->get_viewport()) {
			hcv->get_viewport()->update();
		}

		return false;
	}

	bool keyPressEvent(PVZoomableDrawingArea* zda, QKeyEvent *event) override
	{
		PVHitCountView *hcv = get_hit_count_view(zda);
		switch (event->key()) {
		case Qt::Key_Space:
			if (event->modifiers() == Qt::NoModifier) {
				if(hcv->params_widget()->isHidden()) {
					hcv->params_widget()->setPersistence(false);
					hcv->params_widget()->popup(QCursor::pos(), true);
					return true;
				}
			} else if (event->modifiers() == Qt::ControlModifier) {
				hcv->params_widget()->setPersistence(true);
				hcv->params_widget()->popup(QCursor::pos(), true);
				return true;
			}
		case Qt::Key_Home:
			if (event->modifiers() == Qt::ControlModifier) {
				hcv->set_x_zoom_level_from_sel();

				hcv->reconfigure_view();

				QScrollBar64 *sb = hcv->get_horizontal_scrollbar();
				sb->setValue(0);

				zda->get_viewport()->update();
				zoom_has_changed(zda, PVZoomableDrawingAreaConstraints::X);
			}
			else {
				hcv->reset_view();
				hcv->reconfigure_view();
				hcv->_update_all_timer.start();
			}
			return true;
		case Qt::Key_S:
			if (event->modifiers() == Qt::AltModifier) {
				hcv->toggle_auto_x_zoom_sel();
				return true;
			}
			break;
		case Qt::Key_H:
			if (hcv->help_widget()->isHidden()) {
				hcv->help_widget()->popup(hcv->get_viewport(),
				                          PVWidgets::PVTextPopupWidget::AlignCenter,
				                          PVWidgets::PVTextPopupWidget::ExpandAll, 16);
			}
			break;
		default:
			break;
		}

		return false;
	}

	bool wheelEvent(PVZoomableDrawingArea* zda, QWheelEvent* event)
	{
		int mask = 0;

		if (event->modifiers() == Qt::NoModifier) {
			mask = PVZoomableDrawingAreaConstraints::Y;
		} else if (event->modifiers() == Qt::ControlModifier) {
			mask = PVZoomableDrawingAreaConstraints::X | PVZoomableDrawingAreaConstraints::Y;
		} else if (event->modifiers() == Qt::ShiftModifier) {
			mask = PVZoomableDrawingAreaConstraints::X;
		}

		PVHitCountView *hcv = get_hit_count_view(zda);
		int inc = (event->delta() > 0)?1:-1;

		if (mask & PVZoomableDrawingAreaConstraints::X) {

			event->setAccepted(true);

			if (increment_zoom_value(hcv, mask, inc)) {
				QPointF scene_pos = hcv->map_margined_to_scene(QPointF(0, 0));

				hcv->reconfigure_view();

				int scroll_x = hcv->map_to_view(hcv->map_margined_from_scene(scene_pos)).x();
				hcv->get_horizontal_scrollbar()->setValue(scroll_x);

				hcv->get_viewport()->update();
				zoom_has_changed(hcv, mask);
				return true;
			}
		} else 	if (mask != 0) {
			int inc = (event->delta() > 0)?1:-1;

			event->setAccepted(true);

			if (increment_zoom_value(zda, mask, inc)) {
				hcv->_do_auto_scale = hcv->_auto_x_zoom_sel;
				zda->reconfigure_view();
				zda->get_viewport()->update();
				zoom_has_changed(zda, mask);
				return true;
			}
		}

		return false;
	}

protected:
	static inline PVHitCountView *get_hit_count_view(PVZoomableDrawingArea *zda)
	{
		assert(qobject_cast<PVHitCountView*>(zda));
		return static_cast<PVHitCountView*>(zda);
	}
};


class PVSelectionRectangleHitCountView : public PVSelectionSquare
{
public:
	PVSelectionRectangleHitCountView(PVHitCountView* hcv) :
	PVSelectionSquare(hcv->get_scene()),
	_hcv(hcv)
	{}

protected:
	void commit(bool use_selection_modifiers) override
	{
		QRectF r = _selection_graphics_item->rect();
		Picviz::PVView& view = lib_view();

		unsigned int modifiers = (unsigned int) QApplication::keyboardModifiers();
		modifiers &= ~Qt::KeypadModifier;

		bool use_selectable = true;
		if (use_selection_modifiers && (modifiers == PVSelectionGenerator::AND_MODIFIER || modifiers == PVSelectionGenerator::NAND_MODIFIER)) {
			use_selectable = false;
		}

		PVSelectionGenerator::compute_selection_from_hit_count_view_rect(
			_hcv->get_hit_graph_manager(),
			r, _hcv->get_max_count(),
			view.get_volatile_selection(),
			use_selectable
		);
		PVSelectionGenerator::process_selection(view.shared_from_this(), use_selection_modifiers);
	}

	Picviz::PVView& lib_view() override
	{
		return _hcv->lib_view();
	}

private:
	PVHitCountView* _hcv;
};

}

/*****************************************************************************
 * PVParallelView::PVHitCountView::PVHitCountView
 *****************************************************************************/

PVParallelView::PVHitCountView::PVHitCountView(Picviz::PVView_sp &pvview_sp,
                                               const uint32_t *col_plotted,
                                               const PVRow nrows,
                                               const PVCol axis_index,
                                               QWidget *parent) :
	PVParallelView::PVZoomableDrawingAreaWithAxes(parent),
	_pvview(*pvview_sp),
	_axis_index(axis_index),
	_hit_graph_manager(col_plotted, nrows, 2,
	                   layer_stack_output_selection(),
	                   real_selection()),
	_view_deleted(false),
	_show_bg(true),
	_auto_x_zoom_sel(false),
	_do_auto_scale(false),
	_use_log_color(false)
{
	 set_gl_viewport();

	/* computing the highest scene width to setup it... and do the first
	 * run to initialize the manager's buffers :-)
	 */
	get_hit_graph_manager().change_and_process_view(0, 0, .5);
	_max_count = get_hit_graph_manager().get_max_count_all();

	QRectF r(0, 0, _max_count, 1L << 32);
	set_scene_rect(r);
	get_scene()->setSceneRect(r);

	/* X zoom converter
	 */
	get_x_axis_zoom().set_zoom_converter(&x_zoom_converter());
	get_y_axis_zoom().set_zoom_converter(&y_zoom_converter());

	_sel_rect = new PVSelectionRectangleHitCountView(this);

	/* interactor/constraints
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
	// setMaximumWidth(1024);
	// setMaximumHeight(1024);

	set_alignment(Qt::AlignLeft | Qt::AlignTop);
#if 0
	set_horizontal_scrollbar_policy(Qt::ScrollBarAlwaysOff);
#else
	set_horizontal_scrollbar_policy(Qt::ScrollBarAlwaysOn);
#endif
	set_x_legend("Occurrence count");
	set_y_legend(pvview_sp->get_axis_name(axis_index));
	set_decoration_color(Qt::white);
	set_ticks_per_level(8);

	_params_widget = new PVHitCountViewParamsWidget(this);
	_params_widget->update_widgets();
	_params_widget->hide();

	_update_all_timer.setInterval(RENDER_TIMEOUT);
	_update_all_timer.setSingleShot(true);
	connect(&_update_all_timer, SIGNAL(timeout()),
			this, SLOT(do_update_all()));

	connect(this, SIGNAL(zoom_has_changed(int)),
	        this, SLOT(do_zoom_change(int)));
	connect(this, SIGNAL(pan_has_changed()),
	        this, SLOT(do_pan_change()));

	connect(get_vertical_scrollbar(), SIGNAL(valueChanged(qint64)),
	        this, SLOT(do_pan_change()));

	_help_widget = new PVWidgets::PVTextPopupWidget(this);
	_help_widget->hide();

	_help_widget->initTextFromFile("hit count view's help",
	                               ":help-style");
	_help_widget->addTextFromFile(":help-selection");
	_help_widget->addTextFromFile(":help-layers");
	_help_widget->addTextFromFile(":help-lines");
	_help_widget->newColumn();
	_help_widget->addTextFromFile(":help-view");
	_help_widget->newTable();
	_help_widget->addTextFromFile(":help-shortcuts-hit-count-view");
	_help_widget->newColumn();
	_help_widget->addTextFromFile(":help-mouse-hit-count-view");
	_help_widget->finalizeText();

	// Register view for unselected & zombie lines toggle
	PVHive::PVObserverSignal<bool>* obs = new PVHive::PVObserverSignal<bool>(this);
	PVHive::get().register_observer(pvview_sp,
	                                [=](Picviz::PVView& view) {
		                                return &view.are_view_unselected_zombie_visible();
	                                },
	                                *obs);
	obs->connect_refresh(this, SLOT(toggle_unselected_zombie_visibility()));
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
	QMetaObject::invokeMethod(this, "update_sel", Qt::QueuedConnection);
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::update_all_async
 *****************************************************************************/

void PVParallelView::PVHitCountView::update_all_async()
{
	QMetaObject::invokeMethod(this, "update_all", Qt::QueuedConnection);
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

void PVParallelView::PVHitCountView::set_x_zoom_level_from_sel()
{
	const uint32_t max_count_sel = get_hit_graph_manager().get_max_count_selected();
	if (max_count_sel > 0) {
		set_zoom_value(PVZoomableDrawingAreaConstraints::X, x_zoom_converter().scale_to_zoom(get_margined_viewport_width()/(double)max_count_sel));
	}
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::drawBackground
 *****************************************************************************/

void PVParallelView::PVHitCountView::drawBackground(QPainter *painter,
                                                    const QRectF &margined_rect)
{
	//int img_top = map_margined_from_scene(QPointF(0, _block_base_pos)).y();
	int img_top = map_margined_from_scene(QPointF(0, get_hit_graph_manager().last_y_min())).y();

	int block_view_offset = - img_top;

	int zoom_level = get_y_axis_zoom().get_clamped_value();
	double rel_y_scale = y_zoom_to_scale(zoom_level - _block_zoom_value);

	painter->fillRect(margined_rect, common::color_view_bg());
	painter->setPen(QPen(Qt::white));

	int x_axis_right = std::min((int)map_margined_from_scene(QPointF(_max_count, 0.)).x(),
	                            get_x_axis_length());

	if (show_bg()) {
		// BENCH_START(hcv_draw_all);
		draw_lines(painter,
		           x_axis_right,
		           block_view_offset,
		           rel_y_scale,
		           get_hit_graph_manager().buffer_all(),
		           0);

		// painter->setOpacity(0.25);
		// BENCH_START(hcv_draw_selectable);
		draw_lines(painter,
		           x_axis_right,
		           block_view_offset,
		           rel_y_scale,
		           get_hit_graph_manager().buffer_selectable(),
		           63);
		// BENCH_STOP(hcv_draw_selectable);
		// BENCH_STAT_TIME(hcv_draw_selectable);
	}

	// selected events
	painter->setOpacity(1.0);
	// BENCH_START(hcv_draw_selected);
	draw_lines(painter,
	           x_axis_right,
	           block_view_offset,
	           rel_y_scale,
	           get_hit_graph_manager().buffer_selected(),
	           255);
	// BENCH_STOP(hcv_draw_selected);
	// BENCH_STAT_TIME(hcv_draw_selected);

	draw_decorations(painter, margined_rect);
}

void PVParallelView::PVHitCountView::drawForeground(QPainter* /*painter*/, const QRectF& /*rect*/)
{
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::draw_lines
 *****************************************************************************/

void PVParallelView::PVHitCountView::draw_lines(QPainter *painter,
                                                const int x_max,
                                                const int block_view_offset,
                                                const double rel_y_scale,
                                                const uint32_t *buffer,
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

		vx = PVCore::min(x_max, vx);

		double ratio = count - min_value;

		if (use_log_color()) {
			ratio = log(ratio) / log_range_value;
		} else {
			ratio /= range_value;
		}

		// using fillRect is faster than drawLine... 10 times faster
		// for the color-ramp, it's: ratio * (RED - GREEN) + GREEN
		painter->fillRect(0, y_val, vx, 1, QColor::fromHsv(ratio * (0 - 120) + 120, 255, hsv_value));
	}
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::do_zoom_change
 *****************************************************************************/

void PVParallelView::PVHitCountView::do_zoom_change(int axes)
{
	if (axes & PVZoomableDrawingAreaConstraints::Y) {
		// Hide selection square as it means nothing now.
		_sel_rect->hide();
		if (_do_auto_scale) {
			get_horizontal_scrollbar()->setValue(0);
		}
	}
	_update_all_timer.start();
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::do_pan_change
 *****************************************************************************/

void PVParallelView::PVHitCountView::do_pan_change()
{
	_update_all_timer.start();
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::do_update_all
 *****************************************************************************/

void PVParallelView::PVHitCountView::do_update_all()
{
	int zoom_value, zoom_level;
	double alpha;

	get_hit_graph_manager().set_layer_sel(layer_stack_output_selection());
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
	uint64_t block_size = 1L << (32-zoom_level);
	uint32_t block_y_min = (uint64_t)y_min & ~(block_size - 1);

	get_hit_graph_manager().change_and_process_view(block_y_min, zoom_level, alpha);

	if (_do_auto_scale) {
		set_x_zoom_level_from_sel();

		const ViewportAnchor old_anchor = get_transformation_anchor();
		set_transformation_anchor(PVGraphicsView::NoAnchor);
		reconfigure_view();
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
	get_hit_graph_manager().process_all_buffers();
	get_viewport()->update();
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::update_sel
 *****************************************************************************/

void PVParallelView::PVHitCountView::update_sel()
{
	get_hit_graph_manager().process_buffer_selected();
	if (_auto_x_zoom_sel) {
		set_x_zoom_level_from_sel();
		set_transformation_anchor(PVGraphicsView::NoAnchor);
		reconfigure_view();
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

QString PVParallelView::PVHitCountView::get_y_value_at(const qint64 /*pos*/) const
{
	return QString();
}
