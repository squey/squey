/**
 * \file PVParallelScene.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvparallelview/PVFullParallelScene.h>

#include <tbb/task.h>

#include <QtCore>
#include <QKeyEvent>

#define CRAND() (127 + (random() & 0x7F))

PVParallelView::PVFullParallelScene::PVFullParallelScene(Picviz::FakePVView::shared_pointer view_sp, PVParallelView::PVZonesManager& zm, PVParallelView::PVLinesView::zones_drawing_t::bci_backend_t& bci_backend, tbb::task* root_sel) :
	QGraphicsScene(),
	_lines_view(zm, bci_backend),
	_view_sp(view_sp),
	_parallel_view(new PVFullParallelView(this)),
	_selection_square(new PVParallelView::PVSelectionSquareGraphicsItem(this)),
	_selection_generator(_lines_view.get_zones_manager()),
	_sel(view_sp->get_view_selection()),
	_root_sel(root_sel)
{
	_rendering_job_sel = new PVRenderingJob(this);
	_rendering_job_bg  = new PVRenderingJob(this);
	_rendering_job_all = new PVRenderingJob(this);

	setBackgroundBrush(Qt::black);

	connect(_parallel_view->horizontalScrollBar(), SIGNAL(sliderPressed()), this, SLOT(scrollbar_pressed_Slot()));
	connect(_parallel_view->horizontalScrollBar(), SIGNAL(sliderReleased()), this, SLOT(scrollbar_released_Slot()));
	connect(_selection_square, SIGNAL(commit_volatile_selection()), this, SLOT(commit_volatile_selection_Slot()));

	connect_rendering_job();

	PVParallelView::PVLinesView::list_zone_images_t images = _lines_view.get_zones_images();

	// Add ALL axes
	int pos = 0;
	PVZoneID nzones = (PVZoneID) _lines_view.get_zones_manager().get_number_cols();
	for (PVZoneID z = 0; z < nzones; z++) {
		Picviz::PVAxis* axis = new Picviz::PVAxis();
		axis->set_name(QString("axis ") + QString::number(z));
		axis->set_color(PVCore::PVColor::fromRgba(CRAND(), CRAND(), CRAND(), 0));
		axis->set_titlecolor(PVCore::PVColor::fromRgba(CRAND(), CRAND(), CRAND(), 0));

		if (z < nzones-1) {
			pos = _lines_view.get_zones_manager().get_zone_absolute_pos(z);
		}
		else {
			// Special case for last axis
			pos += _lines_view.get_zones_manager().get_zone_width(z-1);
		}

		PVParallelView::PVAxisGraphicsItem* axisw = new PVParallelView::PVAxisGraphicsItem(axis, z);
		connect(axisw, SIGNAL(axis_sliders_moved(PVZoneID)), this, SLOT(update_selection_from_sliders_Slot(PVZoneID)));
		axisw->setPos(QPointF(pos - PVParallelView::AxisWidth, 0));
		addItem(axisw);
		_axes.push_back(axisw);
		axisw->add_range_sliders(768, 1000);
	}

	_parallel_view->set_total_line_number(_lines_view.get_zones_manager().get_number_rows());

	_heavy_job_timer = new QTimer(this);
	_heavy_job_timer->setInterval(500);
	_heavy_job_timer->setSingleShot(true);
}

PVParallelView::PVFullParallelScene::~PVFullParallelScene()
{
	_rendering_job_sel->deleteLater();
	_rendering_job_bg->deleteLater();
}

void PVParallelView::PVFullParallelScene::connect_rendering_job()
{
	connect(_rendering_job_sel, SIGNAL(zone_rendered(int)), this, SLOT(update_zone_pixmap_sel(int)));
	connect(_rendering_job_bg,  SIGNAL(zone_rendered(int)), this, SLOT(update_zone_pixmap_bg(int)));

	connect(_rendering_job_all, SIGNAL(zone_rendered(int)), this, SLOT(update_zone_pixmap_sel(int)));
	connect(_rendering_job_all, SIGNAL(zone_rendered(int)), this, SLOT(update_zone_pixmap_bg(int)));
}

void PVParallelView::PVFullParallelScene::first_render()
{
	// AG & JBL: FIXME: This must be called after the view has been shown.
	// It seems like a magical QAbstractScrollbarArea stuff, investigation needed...
	PVParallelView::PVLinesView::list_zone_images_t images = _lines_view.get_zones_images();

	// Add visible zones
	_zones.reserve(images.size());
	for (PVZoneID z = 0; z < (PVZoneID) images.size() ; z++) {
			ZoneImages zi;
			zi.sel = addPixmap(QPixmap());
			zi.bg = addPixmap(QPixmap());
			zi.bg->setOpacity(0.25);
			zi.img_tmp_sel = _lines_view.get_zones_drawing()->create_image(PVParallelView::ZoneMaxWidth);
			zi.img_tmp_bg  = _lines_view.get_zones_drawing()->create_image(PVParallelView::ZoneMaxWidth);
			_zones.push_back(zi);
			PVZoneID real_zone = z + _lines_view.get_first_drawn_zone();
			if (real_zone < _lines_view.get_zones_manager().get_number_zones()) {
					zi.setPos(QPointF(_lines_view.get_zone_absolute_pos(real_zone), 0));
			}
	}

	uint32_t view_x = _parallel_view->horizontalScrollBar()->value();
	uint32_t view_width = _parallel_view->width();
	
	connect_draw_zone_sel();
	_lines_view.render_all_zones_all_imgs(view_x, view_width, _sel, _render_tasks_bg, _root_sel, _rendering_job_bg);
}


void PVParallelView::PVFullParallelScene::connect_draw_zone_sel()
{
	disconnect(&_lines_view.get_zones_manager(), 0, this, 0);
	connect(&_lines_view.get_zones_manager(), SIGNAL(filter_by_sel_finished(int, bool)), this, SLOT(draw_zone_sel_Slot(int, bool)), Qt::DirectConnection);
}

void PVParallelView::PVFullParallelScene::update_zones_position(bool update_all /*= true*/)
{
	PVParallelView::PVLinesView::list_zone_images_t images = _lines_view.get_zones_images();
	BENCH_START(update);
	for (PVZoneID zid = _lines_view.get_first_drawn_zone(); zid <= _lines_view.get_last_drawn_zone(); zid++) {
		scale_zone_images(zid);
	}
	BENCH_END(update, "update_zone_pixmap", 1, 1, 1, 1);

	// Update axes position
	PVZoneID nzones = (PVZoneID) _lines_view.get_zones_manager().get_number_cols();
	uint32_t pos = 0;

	PVZoneID z = 1;
	if (! update_all) {
		uint32_t view_x = _parallel_view->horizontalScrollBar()->value();
		z = _lines_view.get_zone_from_scene_pos(view_x) + 1;
	}
	for (; z < nzones; z++) {
		if (z < nzones-1) {
			pos = _lines_view.get_zones_manager().get_zone_absolute_pos(z);
		}
		else {
			// Special case for last axis
			pos += _lines_view.get_zones_manager().get_zone_width(z-1);
		};

		_axes[z]->setPos(QPointF(pos - PVParallelView::AxisWidth, 0));
	}

	update_selection_square();
}

void PVParallelView::PVFullParallelScene::store_selection_square()
{
	PVZoneID& zid1 = _selection_barycenter.zid1;
	PVZoneID& zid2 = _selection_barycenter.zid2;
	double& factor1 = _selection_barycenter.factor1;
	double& factor2 = _selection_barycenter.factor2;

	uint32_t abs_left = _selection_square->rect().topLeft().x();
	uint32_t abs_right = _selection_square->rect().bottomRight().x();

	zid1 = _lines_view.get_zone_from_scene_pos(abs_left);
	uint32_t z1_width = _lines_view.get_zone_width(zid1);
	uint32_t alpha = map_to_axis(zid1, QPointF(abs_left, 0)).x();
	factor1 = (double) alpha / z1_width;

	zid2 = _lines_view.get_zone_from_scene_pos(abs_right);
	uint32_t z2_width = _lines_view.get_zone_width(zid2);
	uint32_t beta = map_to_axis(zid2, QPointF(abs_right, 0)).x();
	factor2 = (double) beta / z2_width;
}

void PVParallelView::PVFullParallelScene::update_selection_square()
{
	PVZoneID zid1 = _selection_barycenter.zid1;
	PVZoneID zid2 = _selection_barycenter.zid2;
	if ((zid1 == PVZONEID_INVALID) || (zid2 == PVZONEID_INVALID)) {
		return;
	}
	double factor1 = _selection_barycenter.factor1;
	double factor2 = _selection_barycenter.factor2;

	uint32_t new_left = _lines_view.get_zones_manager().get_zone_absolute_pos(zid1) + (double) _lines_view.get_zone_width(zid1) * factor1;
	uint32_t new_right = _lines_view.get_zones_manager().get_zone_absolute_pos(zid2) + (double) _lines_view.get_zone_width(zid2) * factor2;
	uint32_t abs_top = _selection_square->rect().topLeft().y();
	uint32_t abs_bottom = _selection_square->rect().bottomRight().y();

	_selection_square->setRect(QRectF(QPointF(new_left, abs_top), QPointF(new_right, abs_bottom)));
}

void PVParallelView::PVFullParallelScene::translate_and_update_zones_position()
{
	uint32_t view_x = _parallel_view->horizontalScrollBar()->value();
	uint32_t view_width = _parallel_view->width();
	_lines_view.translate(view_x, view_width, _sel, _root_sel, _render_tasks_bg, _rendering_job_all);
}

void PVParallelView::PVFullParallelScene::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
	if (event->button() == Qt::RightButton) {
		// Store view position to compute translation
		_translation_start_x = event->scenePos().x();
	}
	else {
		_selection_square_pos = event->scenePos();
	}

	QGraphicsScene::mousePressEvent(event);
}

void PVParallelView::PVFullParallelScene::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
	if (event->buttons() == Qt::RightButton) {
		// Translate viewport
		QScrollBar *hBar = _parallel_view->horizontalScrollBar();
		hBar->setValue(hBar->value() + int(_translation_start_x - event->scenePos().x()));
	}
	else if (!sliders_moving() && event->buttons() == Qt::LeftButton)
	{
		// trace square area
		QPointF top_left(qMin(_selection_square_pos.x(), event->scenePos().x()), qMin(_selection_square_pos.y(), event->scenePos().y()));
		QPointF bottom_right(qMax(_selection_square_pos.x(), event->scenePos().x()), qMax(_selection_square_pos.y(), event->scenePos().y()));

		_selection_square->update_rect(QRectF(top_left, bottom_right));
	}

	QGraphicsScene::mouseMoveEvent(event);
}

void PVParallelView::PVFullParallelScene::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
	if (event->button() == Qt::RightButton) {
		// translate zones
		translate_and_update_zones_position();
	}
	else if (!sliders_moving()) {
		if (_selection_square_pos == event->scenePos()) {
			// Remove selection
			_selection_square->clear_rect();
		}
		commit_volatile_selection_Slot();
	}

	QGraphicsScene::mouseReleaseEvent(event);
}

void PVParallelView::PVFullParallelScene::keyPressEvent(QKeyEvent* event)
{
	if (event->key() == Qt::Key_Space) {
		for (PVZoneID zid = _lines_view.get_first_drawn_zone(); zid <= _lines_view.get_last_drawn_zone(); zid++) {
			update_zone_pixmap_bg(zid);
		}
	}
}

void PVParallelView::PVFullParallelScene::try_to_launch_zoom_job()
{
	if (_render_tasks_sel.is_canceling() ||
	    _render_tasks_bg.is_canceling()) {
		return;
	}

	PVLOG_INFO("try_to_launch_zoom_job: tasks canceled, launch new jobs\n");
	_lines_view.cancel_all_rendering();
	_rendering_job_sel->reset();
	_rendering_job_bg->reset();

	const int view_x = _parallel_view->horizontalScrollBar()->value();

	connect_draw_zone_sel();
	connect_rendering_job();

	// Laucnh task jobs
	BENCH_START(tasks);
	_lines_view.render_all_zones_all_imgs(view_x, _parallel_view->width(), _sel, _render_tasks_bg, _root_sel, _rendering_job_bg);
	BENCH_END(tasks, "task launching", 1, 1, 1, 1);

	disconnect(_heavy_job_timer, NULL, this, NULL);
}

void PVParallelView::PVFullParallelScene::wheelEvent(QGraphicsSceneWheelEvent* event)
{
	int zoom = event->delta() / 2;
	if(zoom < 0) {
		zoom = picviz_max(zoom, -PVParallelView::ZoneMinWidth);
	}

	const QPointF mouse_scene_pt = event->scenePos();
	PVZoneID mouse_zid = _lines_view.get_zone_from_scene_pos(mouse_scene_pt.x());
	// Local zoom
	if (event->modifiers() == Qt::ControlModifier) {
		const PVZoneID zid = mouse_zid;
		//_render_tasks_sel.cancel();
		//_render_tasks_bg.cancel();

		uint32_t z_width = _lines_view.get_zone_width(zid);
		if (_lines_view.set_zone_width(zid, z_width+zoom)) {
			update_zones_position(false);
			_lines_view.render_zone_all_imgs(zid, _sel, _render_tasks_bg, _root_sel, _rendering_job_bg);
		}
		update_zones_position();
	}
	//Global zoom
	else if (event->modifiers() == Qt::NoModifier) {
		disconnect(_heavy_job_timer, NULL, this, NULL);
		disconnect(&_lines_view.get_zones_manager(), 0, this, 0);
		disconnect(_rendering_job_sel, NULL, this, NULL);
		disconnect(_rendering_job_bg, NULL, this, NULL);

		// Get the current zone where the mouse is
		const PVZoneID zmouse = mouse_zid;
		int32_t zone_x = map_to_axis(zmouse, mouse_scene_pt).x();
		int32_t mouse_view_x = _parallel_view->mapFromScene(mouse_scene_pt).x();

		_rendering_job_sel->cancel();
		_rendering_job_bg->cancel();
		_lines_view.cancel_all_rendering();

		_lines_view.set_all_zones_width([=](uint32_t width){ return width+zoom; });
		update_zones_position();

		_render_tasks_sel.cancel();
		_render_tasks_bg.cancel();

		// Compute the new view x coordinate
		//zone_x += zoom;
		int32_t view_x = map_from_axis(zmouse, QPointF(zone_x, 0)).x() - mouse_view_x;

		_parallel_view->horizontalScrollBar()->setValue(view_x);

		connect(_heavy_job_timer, SIGNAL(timeout()), this, SLOT(try_to_launch_zoom_job()));

		if (!_task_waiter.isRunning()) {
			_task_waiter = QtConcurrent::run([&]
				{
					_render_tasks_sel.wait();
					_render_tasks_bg.wait();
					_heavy_job_timer->stop();
					_heavy_job_timer->start();
				});
		}
	}
	event->accept();
}

void PVParallelView::PVFullParallelScene::scale_zone_images(PVZoneID zid)
{
	const PVZoneID img_id = zid-_lines_view.get_first_drawn_zone();
	PVParallelView::PVLinesView::list_zone_images_t& images = _lines_view.get_zones_images();

	PVLinesView::backend_image_p_t& img_sel = images[img_id].sel;
	PVLinesView::backend_image_p_t& img_bg = images[img_id].bg;

	const uint32_t zone_width = _lines_view.get_zone_width(zid);
	{
		PVLinesView::backend_image_p_t& scaled_img = _zones[img_id].img_tmp_bg;
		img_bg->resize_width(*scaled_img, zone_width);
		_zones[img_id].bg->setPixmap(QPixmap::fromImage(scaled_img->qimage()));
	}
	{
		PVLinesView::backend_image_p_t& scaled_img = _zones[img_id].img_tmp_sel;
		img_sel->resize_width(*scaled_img, zone_width);
		_zones[img_id].sel->setPixmap(QPixmap::fromImage(scaled_img->qimage()));
	}

	_zones[img_id].setPos(QPointF(_lines_view.get_zone_absolute_pos(zid), 0));
}

void PVParallelView::PVFullParallelScene::update_zone_pixmap_sel(int zid)
{
	if (!_lines_view.is_zone_drawn(zid)) {
		return;
	}

	PVParallelView::PVLinesView::list_zone_images_t& images = _lines_view.get_zones_images();
	const PVZoneID img_id = zid-_lines_view.get_first_drawn_zone();

	// Check whether the image needs scaling.
	const uint32_t zone_width = _lines_view.get_zone_width(zid);

	PVLinesView::backend_image_p_t& img_sel = images[img_id].sel;

	if ((uint32_t) img_sel->width() != zone_width) {
		return;
	}

	_zones[img_id].sel->setPixmap(QPixmap::fromImage(img_sel->qimage()));
	_zones[img_id].sel->setPos(QPointF(_lines_view.get_zone_absolute_pos(zid), 0));
}

void PVParallelView::PVFullParallelScene::update_zone_pixmap_bg(int zid)
{
	if (!_lines_view.is_zone_drawn(zid)) {
		return;
	}

	PVParallelView::PVLinesView::list_zone_images_t& images = _lines_view.get_zones_images();
	const PVZoneID img_id = zid-_lines_view.get_first_drawn_zone();

	// Check whether the image needs scaling.
	const uint32_t zone_width = _lines_view.get_zone_width(zid);

	PVLinesView::backend_image_p_t& img_bg = images[img_id].bg;

	if ((uint32_t) img_bg->width() != zone_width) {
		return;
	}

	PVLOG_INFO("update_zone_pixmap_bg: zone %d\n", zid);
		
	_zones[img_id].bg->setPixmap(QPixmap::fromImage(img_bg->qimage()));
	_zones[img_id].bg->setPos(QPointF(_lines_view.get_zone_absolute_pos(zid), 0));
}

void PVParallelView::PVFullParallelScene::update_zone_pixmap_bgsel(int zid)
{
	update_zone_pixmap_bg(zid);
	update_zone_pixmap_sel(zid);
}

void PVParallelView::PVFullParallelScene::commit_volatile_selection_Slot()
{
	_selection_square->finished();
	PVZoneID zid = _lines_view.get_zones_manager().get_zone_id(_selection_square->rect().x());
	QRect r = map_to_axis(zid, _selection_square->rect());

	uint32_t nb_selected_lines = _selection_generator.compute_selection_from_rect(zid, r, _sel);
	_parallel_view->set_selected_line_number(nb_selected_lines);

	store_selection_square();

	process_selection();
}

void PVParallelView::PVFullParallelScene::update_selection_from_sliders_Slot(PVZoneID zid)
{
	_selection_square->clear_rect();
	uint32_t nb_select = _selection_generator.compute_selection_from_sliders(zid, _axes[zid]->get_selection_ranges(), _sel);
	_parallel_view->set_selected_line_number(nb_select);

	process_selection();
}

void PVParallelView::PVFullParallelScene::process_selection()
{
	PVHive::call<FUNC(Picviz::FakePVView::process_selection)>(_view_sp);
}

void PVParallelView::PVFullParallelScene::cancel_current_job()
{
	if (_rendering_future.isRunning()) {
		PVLOG_INFO("(launch_job_future) Current job is running.. Cancelling it !\n");
		tbb::tick_count start = tbb::tick_count::now();
		// One job is running.. Ask the job to cancel and wait for it !
		_rendering_job_sel->cancel();
		_rendering_job_bg->cancel();
		_rendering_future.waitForFinished();
		tbb::tick_count end = tbb::tick_count::now();
		PVLOG_INFO("(launch_job_future) Job cancellation done in %0.4f ms.\n", (end-start).seconds()*1000.0);
	}
}

void PVParallelView::PVFullParallelScene::wait_end_current_job()
{
	if (_rendering_future.isRunning()) {
		_rendering_future.waitForFinished();
	}
}

bool PVParallelView::PVFullParallelScene::sliders_moving() const
{
	for (PVAxisGraphicsItem* axis : _axes) {
		if (axis->sliders_moving()) {
			return true;
		}
	}
	return false;
}

void PVParallelView::PVFullParallelScene::scrollbar_pressed_Slot()
{
	_translation_start_x = (qreal) _parallel_view->horizontalScrollBar()->value();
}

void PVParallelView::PVFullParallelScene::scrollbar_released_Slot()
{
	translate_and_update_zones_position();
}

void PVParallelView::PVFullParallelScene::update_new_selection()
{
	// Ask for current selection rendering to be cancelled
	_render_tasks_sel.cancel();
	_rendering_job_sel->cancel();

	_render_tasks_sel.wait();
	_rendering_job_sel->reset();
	_lines_view.cancel_sel_rendering();

	connect_draw_zone_sel();

	const uint32_t view_width = _parallel_view->width();
	_lines_view.update_sel_tree(view_width, _view_sp->get_view_selection(), _root_sel);
}

void PVParallelView::PVFullParallelScene::draw_zone_sel_Slot(int zid, bool changed)
{
	if (!_lines_view.is_zone_drawn(zid)) {
		return;
	}

	if (_render_tasks_sel.is_canceling() || _rendering_job_sel->should_cancel()) {
		return;
	}

	_render_tasks_sel.run([&, zid]
		{
			this->_lines_view.render_zone_sel(zid, this->_rendering_job_sel);
		}
	);
}
