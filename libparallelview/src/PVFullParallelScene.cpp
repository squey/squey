/**
 * \file PVParallelScene.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvparallelview/PVFullParallelScene.h>

#define CRAND() (127 + (random() & 0x7F))

PVParallelView::PVFullParallelScene::PVFullParallelScene(QObject* parent, PVParallelView::PVLinesView* lines_view) :
	QGraphicsScene(parent),
	_lines_view(lines_view),
	_selection_square(new PVParallelView::PVSelectionSquareGraphicsItem(this)),
	_selection_generator(_lines_view->get_zones_manager())
{
	_rendering_job = new PVRenderingJob(this);
	setBackgroundBrush(Qt::black);

	connect(view()->horizontalScrollBar(), SIGNAL(sliderPressed()), this, SLOT(scrollbar_pressed_Slot()));
	connect(view()->horizontalScrollBar(), SIGNAL(sliderReleased()), this, SLOT(scrollbar_released_Slot()));
	connect(_rendering_job, SIGNAL(zone_rendered(int)), this, SLOT(update_zone_pixmap_Slot(int)));
	connect(_selection_square, SIGNAL(commit_volatile_selection()), this, SLOT(commit_volatile_selection_Slot()));

	PVParallelView::PVLinesView::list_zone_images_t images = _lines_view->get_zones_images();

	// Add ALL axes
	int pos = 0;
	PVZoneID nzones = (PVZoneID) _lines_view->get_zones_manager().get_number_cols();
	for (PVZoneID z = 0; z < nzones; z++) {
		Picviz::PVAxis* axis = new Picviz::PVAxis();
		axis->set_name(QString("axis ") + QString::number(z));
		axis->set_color(PVCore::PVColor::fromRgba(CRAND(), CRAND(), CRAND(), 0));
		axis->set_titlecolor(PVCore::PVColor::fromRgba(CRAND(), CRAND(), CRAND(), 0));

		if (z < nzones-1) {
			pos = _lines_view->get_zones_manager().get_zone_absolute_pos(z);
		}
		else {
			// Special case for last axis
			pos += _lines_view->get_zones_manager().get_zone_width(z-1);
		}

		PVParallelView::PVAxisGraphicsItem* axisw = new PVParallelView::PVAxisGraphicsItem(axis, z);
		connect(axisw, SIGNAL(axis_sliders_moved(uint32_t)), this, SLOT(update_selection_Slot(uint32_t)));
		axisw->setPos(QPointF(pos - PVParallelView::AxisWidth, 0));
		addItem(axisw);
		_axes.push_back(axisw);
		axisw->add_range_sliders(768, 1000);
	}

	view()->set_total_line_number(_lines_view->get_zones_manager().get_number_rows());
}

PVParallelView::PVFullParallelScene::~PVFullParallelScene()
{
	_rendering_job->deleteLater();
}

void PVParallelView::PVFullParallelScene::first_render()
{
	// AG & JBL: FIXME: This must be called after the view has been shown.
	// It seems like a magical QAbstractScrollbarArea stuff, investigation needed...
	uint32_t view_x = view()->horizontalScrollBar()->value();
	uint32_t view_width = view()->width();
	_lines_view->render_all(view_x, view_width);

	PVParallelView::PVLinesView::list_zone_images_t images = _lines_view->get_zones_images();

	// Add visible zones
	_zones.reserve(images.size());
	for (PVZoneID z = 0; z < (PVZoneID) images.size() ; z++) {
		ZoneImages zi;
		zi.sel = addPixmap(QPixmap::fromImage(images[z].sel->qimage()));
		zi.bg = addPixmap(QPixmap::fromImage(images[z].bg->qimage()));
		zi.bg->setOpacity(0.25);
		_zones.push_back(zi);
		PVZoneID real_zone = z + _lines_view->get_first_drawn_zone();
		if (real_zone < _lines_view->get_zones_manager().get_number_zones()) {
			zi.setPos(QPointF(_lines_view->get_zone_absolute_pos(real_zone), 0));
		}
	}
}

void PVParallelView::PVFullParallelScene::update_zones_position(bool update_all /*= true*/)
{
	PVParallelView::PVLinesView::list_zone_images_t images = _lines_view->get_zones_images();
	for (PVZoneID zid = _lines_view->get_first_drawn_zone(); zid <= _lines_view->get_last_drawn_zone(); zid++) {
		update_zone_pixmap_Slot(zid);
	}

	// Update axes position
	PVZoneID nzones = (PVZoneID) _lines_view->get_zones_manager().get_number_cols();
	uint32_t pos = 0;

	PVZoneID z = 1;
	if (! update_all) {
		uint32_t view_x = view()->horizontalScrollBar()->value();
		z = _lines_view->get_zone_from_scene_pos(view_x) + 1;
	}
	for (; z < nzones; z++) {
		if (z < nzones-1) {
			pos = _lines_view->get_zones_manager().get_zone_absolute_pos(z);
		}
		else {
			// Special case for last axis
			pos += _lines_view->get_zones_manager().get_zone_width(z-1);
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

	zid1 = _lines_view->get_zone_from_scene_pos(abs_left);
	uint32_t z1_width = _lines_view->get_zone_width(zid1);
	uint32_t alpha = map_to_axis(zid1, QPointF(abs_left, 0)).x();
	factor1 = (double) alpha / z1_width;

	zid2 = _lines_view->get_zone_from_scene_pos(abs_right);
	uint32_t z2_width = _lines_view->get_zone_width(zid2);
	uint32_t beta = map_to_axis(zid2, QPointF(abs_right, 0)).x();
	factor2 = (double) beta / z2_width;
}

void PVParallelView::PVFullParallelScene::update_selection_square()
{
	PVZoneID zid1 = _selection_barycenter.zid1;
	PVZoneID zid2 = _selection_barycenter.zid2;
	double factor1 = _selection_barycenter.factor1;
	double factor2 = _selection_barycenter.factor2;

	uint32_t new_left = _lines_view->get_zones_manager().get_zone_absolute_pos(zid1) + (double) _lines_view->get_zone_width(zid1) * factor1;
	uint32_t new_right = _lines_view->get_zones_manager().get_zone_absolute_pos(zid2) + (double) _lines_view->get_zone_width(zid2) * factor2;
	uint32_t abs_top = _selection_square->rect().topLeft().y();
	uint32_t abs_bottom = _selection_square->rect().bottomRight().y();

	_selection_square->setRect(QRectF(QPointF(new_left, abs_top), QPointF(new_right, abs_bottom)));
}

void PVParallelView::PVFullParallelScene::translate_and_update_zones_position()
{
	cancel_current_job();
	uint32_t view_x = view()->horizontalScrollBar()->value();
	uint32_t view_width = view()->width();
	launch_job_future([&, view_x, view_width](PVRenderingJob& rendering_job)
		{
			return _lines_view->translate(view_x, view_width, rendering_job);
		}
	);
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
		QScrollBar *hBar = view()->horizontalScrollBar();
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

void PVParallelView::PVFullParallelScene::wheelEvent(QGraphicsSceneWheelEvent* event)
{
	int zoom = event->delta() / 2;
	if(zoom < 0) {
		zoom = picviz_max(zoom, -PVParallelView::ZoneMinWidth);
	}

	const QPointF mouse_scene_pt = event->scenePos();
	PVZoneID mouse_zid = _lines_view->get_zone_from_scene_pos(mouse_scene_pt.x());
	// Local zoom
	if (event->modifiers() == Qt::ControlModifier) {
		const PVZoneID zid = mouse_zid;
		wait_end_current_job();
		uint32_t z_width = _lines_view->get_zone_width(zid);
		if (_lines_view->set_zone_width(zid, z_width+zoom)) {
			update_zones_position(false);
			launch_job_future([&](PVRenderingJob& rendering_job)
				{
				return _lines_view->render_zone_all_imgs(zid, rendering_job);
				}
			);
		}
		update_zones_position();
	}
	//Global zoom
	else if (event->modifiers() == Qt::NoModifier) {
		cancel_current_job();

		// Get the current zone where the mouse is
		const PVZoneID zmouse = mouse_zid;
		int32_t zone_x = map_to_axis(zmouse, mouse_scene_pt).x();
		int32_t mouse_view_x = view()->mapFromScene(mouse_scene_pt).x();

		_lines_view->set_all_zones_width([=](uint32_t width){ return width+zoom; });
		update_zones_position();

		// Compute the new view x coordinate
		//zone_x += zoom;
		int32_t view_x = map_from_axis(zmouse, QPointF(zone_x, 0)).x() - mouse_view_x;

		view()->horizontalScrollBar()->setValue(view_x);

		launch_job_future([&](PVRenderingJob& rendering_job)
		  {
		  return _lines_view->render_all(view_x, view()->width(), rendering_job);
		  }
		);
	}
	event->accept();
}

void PVParallelView::PVFullParallelScene::update_zone_pixmap_Slot(int zid)
{
	if (!_lines_view->is_zone_drawn(zid)) {
		return;
	}
	PVParallelView::PVLinesView::list_zone_images_t& images = _lines_view->get_zones_images();
	const PVZoneID img_id = zid-_lines_view->get_first_drawn_zone();

	// Check whether the image needs scaling.
	QImage qimg_bg = images[img_id].bg->qimage();
	QImage qimg_sel = images[img_id].sel->qimage();
	QImage final_img_bg;
	QImage final_img_sel;
	const uint32_t zone_width = _lines_view->get_zone_width(zid);
	if ((uint32_t) qimg_sel.width() != zone_width) {
		final_img_bg = qimg_bg.scaled(zone_width, qimg_bg.height(), Qt::IgnoreAspectRatio, Qt::FastTransformation);
		final_img_sel = qimg_sel.scaled(zone_width, qimg_sel.height(), Qt::IgnoreAspectRatio, Qt::FastTransformation);
	}
	else {
		final_img_bg = qimg_bg;
		final_img_sel = qimg_sel;
	}

	// Convert the image to a pixmap
	_zones[img_id].setPixmap(QPixmap::fromImage(final_img_sel), QPixmap::fromImage(final_img_bg));
	_zones[img_id].setPos(QPointF(_lines_view->get_zone_absolute_pos(zid), 0));
}

void PVParallelView::PVFullParallelScene::update_selection_Slot(uint32_t axis_id)
{
	_selection_square->clear_rect();
	uint32_t nb_select = _selection_generator.compute_selection_from_sliders(axis_id, _axes[axis_id]->get_selection_ranges(), _sel);
	view()->set_selected_line_number(nb_select);

	cancel_current_job();
	launch_job_future([&](PVRenderingJob& rendering_job)
		{
			return _lines_view->update_sel_from_zone(view()->width(), axis_id, _sel, rendering_job);
		}
	);
}

void PVParallelView::PVFullParallelScene::cancel_current_job()
{
	if (_rendering_future.isRunning()) {
		PVLOG_INFO("(launch_job_future) Current job is running.. Cancelling it !\n");
		tbb::tick_count start = tbb::tick_count::now();
		// One job is running.. Ask the job to cancel and wait for it !
		_rendering_job->cancel();
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
	_translation_start_x = (qreal) view()->horizontalScrollBar()->value();
}

void PVParallelView::PVFullParallelScene::scrollbar_released_Slot()
{
	translate_and_update_zones_position();
}

void PVParallelView::PVFullParallelScene::commit_volatile_selection_Slot()
{
	_selection_square->finished();
	PVZoneID zid = _lines_view->get_zones_manager().get_zone_id(_selection_square->rect().x());
	QRect r = map_to_axis(zid, _selection_square->rect());

	cancel_current_job();
	uint32_t nb_select = _selection_generator.compute_selection_from_rect(zid, r, _sel);
	view()->set_selected_line_number(nb_select);
	launch_job_future([&](PVRenderingJob& rendering_job)
		{
			return _lines_view->update_sel_from_zone(view()->width(), zid, _sel, rendering_job);
		}
	);

	store_selection_square();
}
