
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
	_zt(zt),
	_col_plotted(col_plotted),
	_nrows(nrows),
	_axis_index(axis_index),
	_hit_graph_manager(_zt, _col_plotted, _nrows, 2, pvview_sp->get_real_output_selection()),
	_view_deleted(false)
{
	/* computing the highest scene width to setup it
	 */
	_hit_graph_manager.change_and_process_view(0, 0, .5);

	uint32_t max_value = 0;
	const uint32_t *buffer = _hit_graph_manager.buffer_all();
	for(uint32_t i = 0; i < 1024; ++i) {
		if (max_value < buffer[i]) {
			max_value = buffer[i];
		}
	}

	QRectF r(0, 0, max_value, 1L << 32);
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
	set_x_legend("occurrence count");
	set_y_legend(pvview_sp->get_axis_name(axis_index));

	set_horizontal_scrollbar_policy(Qt::ScrollBarAlwaysOff);

	set_decoration_color(Qt::white);
	set_ticks_count(8);

	_back_image = QImage(1024, 2048,
	                     QImage::Format_ARGB32);
	_back_image_pos = 0;

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
	if (_view_deleted) {
		common::get_lib_view(*_pvview_sp.get())->remove_hit_count_view(this);
	}
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::about_to_be_deleted
 *****************************************************************************/

void PVParallelView::PVHitCountView::about_to_be_deleted()
{
	_view_deleted = true;
	_update_all_timer.stop();
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
	recompute_margins(painter, rect);
	recompute_decorations_geometry();

	int y_axis_length = get_y_axis_length();
	int view_top = rect.height() - (get_scene_bottom_margin() + y_axis_length);
	int img_top = map_from_scene(QPointF(0, _back_image_pos)).y();

	// std::cout << "view_top: " << view_top << std::endl;
	// std::cout << "img_top: " << img_top << std::endl;
	// std::cout << "y_axis_length: " << y_axis_length << std::endl;

	int dy = view_top - img_top;


	QRect viewport_rect = get_real_viewport_rect();

	QRect view_rect = QRect(viewport_rect.x(), view_top,
	                        viewport_rect.width(), y_axis_length);

	QRect subimg_rect;

	if (dy < 0) {
		subimg_rect = QRect(0, 0,
		                    viewport_rect.width(), y_axis_length);
	} else {
		subimg_rect = QRect(0, dy,
		                    viewport_rect.width(), y_axis_length);
	}

	painter->fillRect(rect, QColor::fromRgbF(0.1, 0.1, 0.1, 1.0));

	painter->drawImage(view_rect, _back_image, subimg_rect);

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
	std::cout << "recomputing graph" << std::endl;
	recompute_back_buffer();
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::update_all
 *****************************************************************************/

void PVParallelView::PVHitCountView::update_all()
{
	_hit_graph_manager.process_all();
	recompute_back_buffer();
}

/*****************************************************************************
 * PVParselelView::PVHitCountView::update_sel
 *****************************************************************************/

void PVParallelView::PVHitCountView::update_sel()
{
	_hit_graph_manager.process_sel();
	recompute_back_buffer();
}

/*****************************************************************************
 * PVParallelView::PVHitCountView::recompute_back_buffer
 *****************************************************************************/

void PVParallelView::PVHitCountView::recompute_back_buffer()
{
	int calc_zoom = get_relative_zoom_value() / zoom_steps;
	int calc_steps = get_relative_zoom_value() % zoom_steps;

	uint32_t y_min = map_to_scene(0, get_scene_top_margin()).y();
	uint64_t block_size = 1L << (32-calc_zoom);
	uint32_t block_y_min = (uint64_t)y_min & ~(block_size - 1);

	double alpha = 0.5;
	uint32_t buffer_size = 1024;

	if (get_relative_zoom_value() != 0) {
		alpha = 0.5 * pow(root_step, calc_steps);
		buffer_size = alpha * 2048;
	}

	std::cout << "alpha: " << alpha << std::endl;

	_hit_graph_manager.change_and_process_view(block_y_min, calc_zoom, alpha);

	uint32_t max_value = 0;
	const uint32_t *buffer = _hit_graph_manager.buffer_all();
	for(uint32_t i = 0; i < buffer_size; ++i) {
		if (max_value < buffer[i]) {
			max_value = buffer[i];
		}
	}
	double ratio = 1024. / max_value;

	QPainter ipainter(&_back_image);
	ipainter.setPen(QPen(Qt::white));

	_back_image.fill(Qt::transparent);

	ipainter.setOpacity(.25);
	buffer = _hit_graph_manager.buffer_all();
	for(uint32_t i = 0; i < buffer_size; ++i) {
		ipainter.drawLine(0, i, ceil(buffer[i] * ratio), i);
	}

	ipainter.setOpacity(1.);
	buffer = _hit_graph_manager.buffer_sel();
	for(uint32_t i = 0; i < buffer_size; ++i) {
		ipainter.drawLine(0, i, ceil(buffer[i] * ratio), i);
	}

	_back_image_pos = block_y_min;

	update();
}
