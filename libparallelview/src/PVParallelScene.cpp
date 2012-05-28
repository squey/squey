#include <pvparallelview/PVParallelScene.h>


PVParallelView::PVParallelScene::PVParallelScene(QObject* parent, PVParallelView::PVLinesView* lines_view) :
	QGraphicsScene(parent),
	_lines_view(lines_view),
	_selection_square(new PVParallelView::PVSelectionSquareGraphicsItem(this))
{
	_rendering_job = new PVRenderingJob(this);
	setBackgroundBrush(Qt::black);

	connect(view()->horizontalScrollBar(), SIGNAL(sliderPressed()), this, SLOT(slider_pressed_Slot()));
	connect(view()->horizontalScrollBar(), SIGNAL(sliderReleased()), this, SLOT(slider_released_Slot()));

	_lines_view->render_all_imgs(PVParallelView::ImageWidth);
	PVParallelView::PVLinesView::list_zone_images_t images = _lines_view->get_zones_images();

	// Add visible zones
	int pos = 0;
	_zones.reserve(images.size());
	for (PVZoneID z = 0; z < (PVZoneID) images.size() ; z++) {
		ZoneImages zi;
		zi.sel = addPixmap(QPixmap::fromImage(images[z].sel->qimage()));
		zi.bg = addPixmap(QPixmap::fromImage(images[z].bg->qimage()));
		zi.bg->setOpacity(0.25);
		_zones.push_back(zi);
		if (z < _lines_view->get_zones_manager().get_number_zones()) {
			zi.setPos(QPointF(_lines_view->get_zone_absolute_pos(z), 0));
		}
	}

	// Add ALL axes
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

		PVParallelView::PVAxisWidget* axisw = new PVParallelView::PVAxisWidget(axis);
		axisw->setPos(QPointF(pos - PVParallelView::AxisWidth, 0));
		addItem(axisw);
		_axes.push_back(axisw);
		axisw->add_range_sliders(768, 1000);
	}

	connect(_rendering_job, SIGNAL(zone_rendered(int)), this, SLOT(update_zone_pixmap(int)));
}

void PVParallelView::PVParallelScene::wheelEvent(QGraphicsSceneWheelEvent* event)
{
	int zoom = event->delta() / 2;

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

		//if (_lines_view->set_zone_width_and_render(zid, z_width + zoom)) {
		//}
	}
	//Global zoom
	else
	{
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
