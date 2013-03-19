
#include <picviz/PVView.h>

#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVHitCountView.h>
#include <pvparallelview/PVHitGraphData.h>
#include <pvparallelview/PVZoneTree.h>

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

	/* view configuration
	 */
	set_alignment(Qt::AlignLeft | Qt::AlignTop);

	set_zoom_range(-110, 30);
	set_zoom_value(-110);

	// setMaximumWidth(1024);
	// setMaximumHeight(1024);

	set_alignment(Qt::AlignLeft);
	set_zoom_policy(PVZoomableDrawingArea::AlongY);
	set_pan_policy(PVZoomableDrawingArea::AlongY);
	// set_zoom_policy(PVZoomableDrawingArea::AlongBoth);
	// set_pan_policy(PVZoomableDrawingArea::AlongBoth);
	set_x_legend("occurrence count");
	set_y_legend(pvview_sp->get_axis_name(axis_index));

	set_horizontal_scrollbar_policy(Qt::ScrollBarAlwaysOff);

	set_decoration_color(Qt::white);
	set_ticks_count(8);

	_block_zoom_level = get_zoom_value();

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
 * PVParallelView::PVHitCountView::zoom_to_scale
 *****************************************************************************/

qreal PVParallelView::PVHitCountView::zoom_to_scale(const int zoom_value) const
{
	return pow(2, zoom_value / zoom_steps) * pow(root_step, zoom_value % zoom_steps);
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::scale_to_zoom
 *****************************************************************************/

int PVParallelView::PVHitCountView::scale_to_zoom(const qreal scale_value) const
{
	// non simplified formula is: log2(1/scale_value) / log2(root_steps)
	return -zoom_steps * log2(scale_value);
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::scale_to_transform
 *****************************************************************************/

QTransform PVParallelView::PVHitCountView::scale_to_transform(const qreal x_scale_value,
                                                              const qreal y_scale_value) const
{
	QTransform transfo;
	if (_show_bg) {
		transfo = PVZoomableDrawingAreaWithAxes::scale_to_transform(x_scale_value,
		                                                            y_scale_value);
	} else {
		QRectF v = get_real_viewport_rect();
		QRectF s = get_scene_rect();
		qreal ratio = (v.width() / s.width()) / (v.height() / s.height());

		transfo.scale(x_scale_value * ratio, y_scale_value);
	}
	return transfo;
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::get_y_value_at
 *****************************************************************************/

QString PVParallelView::PVHitCountView::get_y_value_at(const qint64 pos) const
{
	return QString::number(- pos);
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

	int zoom_level = get_zoom_value();
	double rel_scale = zoom_to_scale(zoom_level - _block_zoom_level);

	painter->fillRect(rect, QColor::fromRgbF(0.1, 0.1, 0.1, 1.0));
	painter->setPen(QPen(Qt::white));

	if(_show_bg) {
		painter->setOpacity(0.25);
		draw_lines(painter, src_x, view_top, dy, ratio, rel_scale,
		           _hit_graph_manager.buffer_bg());

		painter->setOpacity(1.0);
		draw_lines(painter, src_x, view_top, dy, ratio, rel_scale,
		           _hit_graph_manager.buffer_sel());
	} else {
		painter->setOpacity(0.25);
		draw_clamped_lines(painter,
		                   map_from_scene(QPointF(0., 0.)).x(),
		                   map_from_scene(QPointF(_max_count, 0.)).x(),
		                   src_x, view_top, dy, rel_scale,
		                   _hit_graph_manager.buffer_bg());
		painter->setOpacity(1.0);
		draw_clamped_lines(painter,
		                   map_from_scene(QPointF(0., 0.)).x(),
		                   map_from_scene(QPointF(_max_count, 0.)).x(),
		                   src_x, view_top, dy, rel_scale,
		                   _hit_graph_manager.buffer_sel());
	}

	draw_decorations(painter, rect);
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::resizeEvent
 *****************************************************************************/

void PVParallelView::PVHitCountView::resizeEvent(QResizeEvent *event)
{
	PVParallelView::PVZoomableDrawingAreaWithAxes::resizeEvent(event);
	_update_all_timer.start();
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::keyPressEvent
 *****************************************************************************/

void PVParallelView::PVHitCountView::keyPressEvent(QKeyEvent *event)
{
	if (event->key() == Qt::Key_Tab) {
		_show_bg = !_show_bg;
		if (_show_bg) {
			set_horizontal_scrollbar_policy(Qt::ScrollBarAlwaysOff);
			set_pan_policy(PVZoomableDrawingArea::AlongY);

		} else {
			set_horizontal_scrollbar_policy(Qt::ScrollBarAlwaysOn);
			set_pan_policy(PVZoomableDrawingArea::AlongBoth);
		}

		update_zoom();
		update_all();
	} else {
		PVZoomableDrawingAreaWithAxes::keyPressEvent(event);
	}
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::draw_lines
 *****************************************************************************/

void PVParallelView::PVHitCountView::draw_lines(QPainter *painter,
                                                const int src_x,
                                                const int view_top,
                                                const int offset,
                                                const double &ratio,
                                                const double rel_scale,
                                                const uint32_t *buffer)
{
	int y_axis_length = get_y_axis_length();

	const int count = (get_relative_zoom_value() == 0)?1024:(2048*_hit_graph_manager.nblocks());

	for (int y = 0; y < count; ++y) {
		const uint32_t v = buffer[y];
		if (v == 0) {
			continue;
		}

		int y_val = (rel_scale * y) - offset;
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
                                                        const double rel_scale,
                                                        const uint32_t *buffer)
{
	int y_axis_length = get_y_axis_length();

	const int count = (get_relative_zoom_value() == 0)?1024:(2048*_hit_graph_manager.nblocks());

	for (int y = 0; y < count; ++y) {
		const uint32_t v = buffer[y];
		if (v == 0) {
			continue;
		}

		int y_val = (rel_scale * y) - offset;
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
	int calc_zoom = get_relative_zoom_value() / zoom_steps;
	int calc_steps = get_relative_zoom_value() % zoom_steps;

	uint32_t y_min = map_to_scene(0, get_scene_top_margin()).y();
	uint64_t block_size = 1L << (32-calc_zoom);
	uint32_t block_y_min = (uint64_t)y_min & ~(block_size - 1);

	double alpha = 0.5;

	if (get_relative_zoom_value() != 0) {
		alpha = 0.5 * pow(root_step, calc_steps);
	}

	BENCH_START(hcv_data_compute);
	_hit_graph_manager.change_and_process_view(block_y_min, calc_zoom, alpha);
	BENCH_STOP(hcv_data_compute);
	BENCH_STAT_TIME(hcv_data_compute);

	_block_base_pos = block_y_min;
	_block_zoom_level = get_zoom_value();

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
