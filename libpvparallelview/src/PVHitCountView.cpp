
#include <picviz/PVView.h>

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

#include <QPainter>
#include <QResizeEvent>
#include <QGraphicsScene>
#include <QScrollBar64>

/**
 * @todo "optical" zoom does not work
 * @todo how do we want to use that view
 */
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

class PVHitCountViewInteractor : public PVZoomableDrawingAreaInteractor
{
public:
	PVHitCountViewInteractor(PVWidgets::PVGraphicsView* parent = nullptr) :
		PVZoomableDrawingAreaInteractor(parent)
	{}

	bool resizeEvent(PVZoomableDrawingArea* zda, QResizeEvent *event) override
	{
		PVHitCountView *hcv = get_hit_count_view(zda);
		hcv->set_x_axis_zoom();

		hcv->reconfigure_view();
		if (hcv->get_viewport()) {
			hcv->get_viewport()->update();
		}

		return false;
	}

	bool  keyPressEvent(PVZoomableDrawingArea* zda, QKeyEvent *event) override
	{
		PVHitCountView *hcv = get_hit_count_view(zda);
		switch (event->key()) {
		case Qt::Key_Home: {
			if (event->modifiers() == Qt::ControlModifier) {
				hcv->set_x_zoom_level_from_sel();

				qreal xpos = hcv->map_margined_to_scene(QPoint(0, 0)).x();

				hcv->reconfigure_view();

				int screen_pos = hcv->map_margined_from_scene(QPointF(xpos, 0)).x();
				QScrollBar64 *sb = hcv->get_horizontal_scrollbar();
				sb->setValue(sb->value() + screen_pos);

				zda->get_viewport()->update();
				zoom_has_changed(zda, PVZoomableDrawingAreaConstraints::X);
			}
			else {
				hcv->reset_view();
				hcv->reconfigure_view();
				hcv->_update_all_timer.start();
			}
			break;
		}
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

		if (mask & PVZoomableDrawingAreaConstraints::X) {
			PVHitCountView *hcv = get_hit_count_view(zda);
			int inc = (event->delta() > 0)?1:-1;

			event->setAccepted(true);

			if (increment_zoom_value(hcv, mask, inc)) {
				qreal xpos = hcv->map_margined_to_scene(QPoint(0, 0)).x();

				hcv->reconfigure_view();

				int screen_pos = hcv->map_margined_from_scene(QPointF(xpos, 0)).x();
				QScrollBar64 *sb = hcv->get_horizontal_scrollbar();
				sb->setValue(sb->value() + screen_pos);

				hcv->get_viewport()->update();
				zoom_has_changed(hcv, mask);
			}
		} else 	if (mask != 0) {
			int inc = (event->delta() > 0)?1:-1;

			event->setAccepted(true);

			if (increment_zoom_value(zda, mask, inc)) {
				zda->reconfigure_view();
				zda->get_viewport()->update();
				zoom_has_changed(zda, mask);
			}
		}

		return event->isAccepted();
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
		PVSelectionGenerator::compute_selection_from_hit_count_view_rect(_hcv->get_hit_graph_manager(),
		                                                                 r, _hcv->get_max_count(),
		                                                                 view.get_volatile_selection());
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

PVParallelView::PVHitCountView::PVHitCountView(const Picviz::PVView_sp &pvview_sp,
                                               const uint32_t *col_plotted,
                                               const PVRow nrows,
                                               const PVCol axis_index,
                                               QWidget *parent) :
	PVParallelView::PVZoomableDrawingAreaWithAxes(parent),
	_pvview(*pvview_sp),
	_axis_index(axis_index),
	_hit_graph_manager(col_plotted, nrows, 2, pvview_sp->get_real_output_selection()),
	_view_deleted(false),
	_show_bg(true)
{
	set_gl_viewport();

	/* computing the highest scene width to setup it
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
	set_x_legend("occurrence count");
	set_y_legend(pvview_sp->get_axis_name(axis_index));
	set_decoration_color(Qt::white);
	set_ticks_per_level(8);

	_update_all_timer.setInterval(150);
	_update_all_timer.setSingleShot(true);
	connect(&_update_all_timer, SIGNAL(timeout()),
			this, SLOT(do_update_all()));

	connect(this, SIGNAL(zoom_has_changed(int)),
	        this, SLOT(do_zoom_change(int)));
	connect(this, SIGNAL(pan_has_changed()),
	        this, SLOT(do_pan_change()));

	connect(get_vertical_scrollbar(), SIGNAL(valueChanged(qint64)),
	        this, SLOT(do_pan_change()));
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
	const int32_t x_zoom_min = get_x_zoom_min();
	get_x_axis_zoom().set_range(x_zoom_min, x_zoom_converter().scale_to_zoom(get_margined_viewport_width()));
	get_x_axis_zoom().set_default_value(x_zoom_min);
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
	_block_zoom_level = get_y_axis_zoom().get_clamped_value();
}

void PVParallelView::PVHitCountView::set_x_zoom_level_from_sel()
{
	const uint32_t max_count_sel = get_hit_graph_manager().get_max_count_sel();
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
	int view_top = 0;
	int img_top = map_margined_from_scene(QPointF(0, _block_base_pos)).y();

	int dy = view_top - img_top;

	int zoom_level = get_y_axis_zoom().get_clamped_value();
	double rel_y_scale = y_zoom_to_scale(zoom_level - _block_zoom_level);

	painter->fillRect(margined_rect, QColor::fromRgbF(0.1, 0.1, 0.1, 1.0));
	painter->setPen(QPen(Qt::white));

	int x_axis_left = map_margined_from_scene(QPointF(0., 0.)).x();
	int x_axis_right = std::min((int)map_margined_from_scene(QPointF(_max_count, 0.)).x(),
								get_x_axis_length());

	painter->setOpacity(0.25);
	draw_clamped_lines(painter,
					   x_axis_left,
					   x_axis_right,
					   view_top, dy, rel_y_scale,
					   get_hit_graph_manager().buffer_bg());
	painter->setOpacity(1.0);
	draw_clamped_lines(painter,
					   x_axis_left,
					   x_axis_right,
					   view_top, dy, rel_y_scale,
					   get_hit_graph_manager().buffer_sel());

	PVParallelView::PVZoomableDrawingAreaWithAxes::drawBackground(painter, margined_rect);
}

void PVParallelView::PVHitCountView::drawForeground(QPainter *painter, const QRectF &rect)
{
	painter->save();
	painter->resetTransform();

	QString txt(QString("Max all: %1 / Max sel: %2").arg(_max_count).arg(get_hit_graph_manager().get_max_count_sel()));
	painter->drawText(QPointF(10, 10), txt);
	painter->restore();
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::draw_lines
 *****************************************************************************/

void PVParallelView::PVHitCountView::draw_lines(QPainter *painter,
                                                const int src_x,
                                                const int view_top,
                                                const int offset,
                                                const double &ratio,
                                                const double rel_y_scale,
                                                const uint32_t *buffer)
{
	const int y_axis_length = get_y_axis_length();
	const int count = get_hit_graph_manager().size_int();

	for (int y = 0; y < count; ++y) {
		const uint32_t v = buffer[y];
		if (v == 0) {
			continue;
		}

		int y_val = (rel_y_scale * y) - offset;
		if ((y_val < 0) || (y_val >= y_axis_length)) {
			continue;
		}

		y_val += view_top;

		int dst_x = src_x + ceil(v * ratio);

		painter->drawLine(src_x, y_val, dst_x, y_val);
	}
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::draw_clamped_lines
 *****************************************************************************/

void PVParallelView::PVHitCountView::draw_clamped_lines(QPainter *painter,
                                                        const int x_min,
                                                        const int x_max,
                                                        const int view_top,
                                                        const int offset,
                                                        const double rel_y_scale,
                                                        const uint32_t *buffer)
{
	const int y_axis_length = get_y_axis_length();
	const int count = get_hit_graph_manager().size_int();

	for (int y = 0; y < count; ++y) {
		const uint32_t v = buffer[y];
		if (v == 0) {
			continue;
		}

		int y_val = (rel_y_scale * y) - offset;
		if ((y_val < 0) || (y_val >= y_axis_length)) {
			continue;
		}

		y_val += view_top;

		int vx = map_margined_from_scene(QPointF(v, 0.)).x();

		if (vx < 0) {
			continue;
		}

		vx = PVCore::min(x_max, vx);

		painter->drawLine(0, y_val, vx, y_val);
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
	// std::cout << "##################################################" << std::endl;
	// std::cout << "# x zoom: " << get_x_axis_zoom().get_clamped_value() << std::endl;
	// std::cout << "# y zoom: " << get_y_axis_zoom().get_clamped_value() << std::endl;

	int rel_zoom = get_y_axis_zoom().get_clamped_relative_value();
	int calc_zoom = rel_zoom / zoom_steps;

	uint32_t y_min = map_margined_to_scene(0, 0).y();
	uint64_t block_size = 1L << (32-calc_zoom);
	uint32_t block_y_min = (uint64_t)y_min & ~(block_size - 1);

	double alpha = 0.5 * y_zoom_converter().zoom_to_scale_decimal(rel_zoom);

	// BENCH_START(hcv_data_compute);
	get_hit_graph_manager().change_and_process_view(block_y_min, calc_zoom, alpha);
	// BENCH_STOP(hcv_data_compute);
	// BENCH_STAT_TIME(hcv_data_compute);

	_block_base_pos = block_y_min;
	_block_zoom_level = get_y_axis_zoom().get_clamped_value();

	get_viewport()->update();
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::update_all
 *****************************************************************************/

void PVParallelView::PVHitCountView::update_all()
{
	get_hit_graph_manager().process_all();
	get_viewport()->update();
}

/*****************************************************************************
 * PVParselelView::PVHitCountView::update_sel
 *****************************************************************************/

void PVParallelView::PVHitCountView::update_sel()
{
	get_hit_graph_manager().process_sel();
	get_viewport()->update();
}
