
#include <picviz/PVView.h>

#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVHitCountView.h>
#include <pvparallelview/PVHitGraphData.h>
#include <pvparallelview/PVZoneTree.h>

#include <pvparallelview/PVZoomableDrawingAreaInteractor.h>
#include <pvparallelview/PVZoomableDrawingAreaInteractorMajorY.h>
#include <pvparallelview/PVZoomableDrawingAreaConstraintsMajorY.h>

#include <QPainter>
#include <QResizeEvent>
#include <QGraphicsScene>
#include <QScrollBar64>

#define print_rect(R) __print_rect(#R, R)

template <typename R>
void __print_rect(const char *text, const R &r)
{
	std::cout << text << ": "
	          << r.x() << " " << r.y() << ", "
	          << r.width() << " " << r.height()
	          << std::endl;
}

namespace PVParallelView
{

template <int STEPS>
class PVHitCountViewZoomConverter : public PVZoomConverterScaledPowerOfTwo<STEPS>
{
public:
	PVHitCountViewZoomConverter(const qreal s) :
		_scale_factor(s)
	{}

	int scale_to_zoom(const qreal value) const override
	{
		return PVZoomConverterScaledPowerOfTwo<STEPS>::scale_to_zoom(value / _scale_factor);
	}

	qreal zoom_to_scale(const int value) const override
	{
		return PVZoomConverterScaledPowerOfTwo<STEPS>::zoom_to_scale(value) * _scale_factor;
	}

private:
	qreal _scale_factor;
};

class PVHitCountViewInteractor : public PVZoomableDrawingAreaInteractor
{
public:
	PVHitCountViewInteractor(PVWidgets::PVGraphicsView* parent = nullptr) :
		PVZoomableDrawingAreaInteractor(parent)
	{}

	bool resizeEvent(PVZoomableDrawingArea* zda, QResizeEvent *event) override
	{
		get_hit_count_view(zda)->_update_all_timer.start();

		return false;
	}

	bool  keyPressEvent(PVZoomableDrawingArea* zda, QKeyEvent *event) override
	{
		if (event->key() == Qt::Key_Tab) {
			PVHitCountView *hcv = get_hit_count_view(zda);

			hcv->_show_bg ^= true;
			if (hcv->_show_bg) {
				hcv->set_horizontal_scrollbar_policy(Qt::ScrollBarAlwaysOff);
			} else {
				hcv->set_horizontal_scrollbar_policy(Qt::ScrollBarAlwaysOn);
			}

			hcv->reconfigure_view();
			hcv->update_all();

			return true;
		} else if(event->key() == Qt::Key_Home) {
			get_hit_count_view(zda)->reset_view();
			return true;
		}

		return false;
	}

protected:
	static inline PVHitCountView *get_hit_count_view(PVZoomableDrawingArea *zda)
	{
		return static_cast<PVHitCountView*>(zda);
	}
};

}

/*****************************************************************************
 * PVParallelView::PVHitCountView::PVHitCountView
 *****************************************************************************/

PVParallelView::PVHitCountView::PVHitCountView(const Picviz::PVView_sp &pvview_sp,
                                               const PVZoneTree &zt,
                                               const uint32_t *col_plotted,
                                               const PVRow nrows,
                                               const PVCol axis_index,
                                               QWidget *parent) :
	PVParallelView::PVZoomableDrawingAreaWithAxes(parent),
	_pvview_sp(pvview_sp),
	_axis_index(axis_index),
	_hit_graph_manager(zt, col_plotted, nrows, 2, pvview_sp->get_real_output_selection()),
	_view_deleted(false),
	_show_bg(true)
{
	/* computing the highest scene width to setup it
	 */
	_hit_graph_manager.change_and_process_view(0, 0, .5);

	_max_count = 0;
	const uint32_t *buffer = _hit_graph_manager.buffer_bg();
	for(uint32_t i = 0; i < 1024; ++i) {
		if (_max_count < buffer[i]) {
			_max_count = buffer[i];
		}
	}

	QRectF r(0, 0, _max_count, 1L << 32);
	set_scene_rect(r);
	get_scene()->setSceneRect(r);

	/* interactor/constraints
	 */
	PVWidgets::PVGraphicsViewInteractorBase *inter =
		declare_interactor<PVZoomableDrawingAreaInteractorMajorY>();
	register_front_all(inter);
	inter = declare_interactor<PVHitCountViewInteractor>();
	register_front_one(QEvent::Resize, inter);
	register_front_one(QEvent::KeyPress, inter);
	install_default_scene_interactor();

	set_constraints(new PVParallelView::PVZoomableDrawingAreaConstraintsMajorY());

	/* X zoom converter
	 */
	_x_zoom_converter = new PVHitCountViewZoomConverter<zoom_steps>(r.height() / r.width());

	/* PVAxisZoom
	 */
	get_x_axis_zoom().set_range(-110, 30);
	get_x_axis_zoom().set_default_value(-110);
	get_x_axis_zoom().set_zoom_converter(_x_zoom_converter);
	get_y_axis_zoom().set_range(-110, 30);
	get_y_axis_zoom().set_default_value(-110);
	get_y_axis_zoom().set_zoom_converter(&_y_zoom_converter);
	reset_view();

	/* view configuration
	 */
	// setMaximumWidth(1024);
	// setMaximumHeight(1024);

	set_alignment(Qt::AlignLeft | Qt::AlignTop);
	set_horizontal_scrollbar_policy(Qt::ScrollBarAlwaysOff);

	set_x_legend("occurrence count");
	set_y_legend(pvview_sp->get_axis_name(axis_index));
	set_decoration_color(Qt::white);
	set_ticks_count(8);

	_block_zoom_level = get_y_axis_zoom().get_clamped_value();

	_update_all_timer.setInterval(150);
	_update_all_timer.setSingleShot(true);
	connect(&_update_all_timer, SIGNAL(timeout()),
			this, SLOT(do_update_all()));

	connect(this, SIGNAL(zoom_has_changed()),
	        this, SLOT(do_zoom_change()));
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
		common::get_lib_view(*_pvview_sp.get())->remove_hit_count_view(this);
	}

	delete _x_zoom_converter;
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::about_to_be_deleted
 *****************************************************************************/

void PVParallelView::PVHitCountView::about_to_be_deleted()
{
	_view_deleted = true;
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
 * PVParallelView::PVHitCountView::get_y_value_at
 *****************************************************************************/

QString PVParallelView::PVHitCountView::get_y_value_at(const qint64 pos) const
{
	return QString::number(- pos);
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::reset_view
 *****************************************************************************/

void PVParallelView::PVHitCountView::reset_view()
{
	set_zoom_value(PVZoomableDrawingAreaConstraints::X
	               | PVZoomableDrawingAreaConstraints::Y,
	               -110);
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::drawBackground
 *****************************************************************************/

void PVParallelView::PVHitCountView::drawBackground(QPainter *painter,
                                                    const QRectF &rect)
{
	recompute_decorations(painter, rect);

	int y_axis_length = get_y_axis_length();
	int view_top = rect.height() - (get_scene_bottom_margin() + y_axis_length);
	int img_top = map_from_scene(QPointF(0, _block_base_pos)).y();

	int dy = view_top - img_top;

	int src_x = get_scene_left_margin();
	double ratio = get_x_axis_length() / (double)_max_count;

	int zoom_level = get_y_axis_zoom().get_clamped_value();
	double rel_y_scale = y_zoom_to_scale(zoom_level - _block_zoom_level);

	painter->fillRect(rect, QColor::fromRgbF(0.1, 0.1, 0.1, 1.0));
	painter->setPen(QPen(Qt::white));

	if(_show_bg) {
		painter->setOpacity(0.25);
		draw_lines(painter, src_x, view_top, dy, ratio, rel_y_scale,
		           _hit_graph_manager.buffer_bg());

		painter->setOpacity(1.0);
		draw_lines(painter, src_x, view_top, dy, ratio, rel_y_scale,
		           _hit_graph_manager.buffer_sel());
	} else {
		painter->setOpacity(0.25);
		draw_clamped_lines(painter,
		                   map_from_scene(QPointF(0., 0.)).x(),
		                   map_from_scene(QPointF(_max_count, 0.)).x(),
		                   src_x, view_top, dy, rel_y_scale,
		                   _hit_graph_manager.buffer_bg());
		painter->setOpacity(1.0);
		draw_clamped_lines(painter,
		                   map_from_scene(QPointF(0., 0.)).x(),
		                   map_from_scene(QPointF(_max_count, 0.)).x(),
		                   src_x, view_top, dy, rel_y_scale,
		                   _hit_graph_manager.buffer_sel());
	}

	draw_decorations(painter, rect);
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
	const int count = (get_y_axis_zoom().get_clamped_relative_value() == 0)?1024:(2048*_hit_graph_manager.nblocks());
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
                                                        const int src_x,
                                                        const int view_top,
                                                        const int offset,
                                                        const double rel_y_scale,
                                                        const uint32_t *buffer)
{
	const int y_axis_length = get_y_axis_length();
	const int count = (get_y_axis_zoom().get_clamped_relative_value() == 0)?1024:(2048*_hit_graph_manager.nblocks());

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

		int vx = map_from_scene(QPointF(v, 0.)).x();

		if (vx < src_x) {
			continue;
		}

		vx = PVCore::min(x_max, vx);

		painter->drawLine(src_x, y_val, vx, y_val);
	}
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::do_zoom_change
 *****************************************************************************/

void PVParallelView::PVHitCountView::do_zoom_change()
{
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
	int rel_zoom = get_y_axis_zoom().get_clamped_relative_value();
	int calc_zoom = rel_zoom / zoom_steps;

	uint32_t y_min = map_to_scene(0, get_scene_top_margin()).y();
	uint64_t block_size = 1L << (32-calc_zoom);
	uint32_t block_y_min = (uint64_t)y_min & ~(block_size - 1);

	double alpha = 0.5 * _y_zoom_converter.zoom_to_scale_decimal(rel_zoom);

	BENCH_START(hcv_data_compute);
	_hit_graph_manager.change_and_process_view(block_y_min, calc_zoom, alpha);
	BENCH_STOP(hcv_data_compute);
	BENCH_STAT_TIME(hcv_data_compute);

	_block_base_pos = block_y_min;
	_block_zoom_level = get_y_axis_zoom().get_clamped_value();

	update();
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::update_all
 *****************************************************************************/

void PVParallelView::PVHitCountView::update_all()
{
	_hit_graph_manager.process_all();
	update();
}

/*****************************************************************************
 * PVParselelView::PVHitCountView::update_sel
 *****************************************************************************/

void PVParallelView::PVHitCountView::update_sel()
{
	_hit_graph_manager.process_sel();
	update();
}
