#ifndef __PVPARALLELSCENE_h__
#define __PVPARALLELSCENE_h__

#include <QGraphicsScene>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsSceneWheelEvent>

#include <picviz/PVAxis.h>
#include <pvparallelview/PVSelectionSquareGraphicsItem.h>
#include <pvparallelview/PVAxisWidget.h>
#include <pvparallelview/PVFullParallelView.h>
#include <pvparallelview/PVLinesView.h>

#include <tbb/tick_count.h>

#define CRAND() (127 + (random() & 0x7F))

namespace PVParallelView {

class PVParallelScene : public QGraphicsScene
{
	Q_OBJECT
public:
	PVParallelScene(QObject* parent, PVParallelView::PVLinesView* lines_view) :
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
			QGraphicsPixmapItem* zone_image = addPixmap(QPixmap::fromImage(images[z].bg->qimage()));
			//zone_image->setOpacity(0.5);
			_zones.push_back(zone_image);
			if (z < _lines_view->get_zones_manager().get_number_zones()) {
				zone_image->setPos(QPointF(_lines_view->get_zone_absolute_pos(z), 0));
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

	virtual ~PVParallelScene()
	{
		_rendering_job->deleteLater();
	}

	inline PVLinesView* get_lines_view() { return _lines_view; }

private:
	PVParallelView::PVFullParallelView* view()
	{
		return (PVParallelView::PVFullParallelView*) parent();
	}


	void update_zones_position(bool update_all = true)
	{
		PVParallelView::PVLinesView::list_zone_images_t images = _lines_view->get_zones_images();
		for (PVZoneID zid = _lines_view->get_first_drawn_zone(); zid <= _lines_view->get_last_drawn_zone(); zid++) {
			update_zone_pixmap(zid);
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
	}

	void mouseMoveEvent(QGraphicsSceneMouseEvent *event)
	{
		if (event->buttons() == Qt::RightButton) {
			// Translate viewport
			QScrollBar *hBar = view()->horizontalScrollBar();
			hBar->setValue(hBar->value() + int(_translation_start_x - event->scenePos().x()));
		}
		else
		{
			// trace square area
			QPointF top_left(qMin(_selection_square_pos.x(), event->scenePos().x()), qMin(_selection_square_pos.y(), event->scenePos().y()));
			QPointF bottom_right(qMax(_selection_square_pos.x(), event->scenePos().x()), qMax(_selection_square_pos.y(), event->scenePos().y()));
			_selection_square->setRect(QRectF(top_left, bottom_right));
		}
	}

	void mousePressEvent(QGraphicsSceneMouseEvent *event)
	{
		if (event->button() == Qt::RightButton)
		{
			// Store view position to compute translation
			_translation_start_x = event->scenePos().x();
		}
		else
		{
			_selection_square_pos = event->scenePos();
		}
	}

	void mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
	{
		if (event->button() == Qt::RightButton)
		{
			// translate zones
			translate_and_update_zones_position();
		}
		else
		{
			PVZoneID zid = _lines_view->get_zones_manager().get_zone_id(_selection_square->rect().x());
			QRect r = map_to_axis(zid, _selection_square->rect());

			_selection_square->compute_selection(zid, r);

			// Remove selection
			if (_selection_square_pos == event->scenePos()) {
				_selection_square->setRect(0, 0, 0, 0);
			}
		}
	}

	void wheelEvent(QGraphicsSceneWheelEvent* event)
	{
		int zoom = event->delta() / 2;

		// Local zoom
		if (event->modifiers() == Qt::ControlModifier) {
			wait_end_current_job();
			PVZoneID zid = _lines_view->get_zone_from_scene_pos(event->scenePos().x());
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
			uint32_t view_x = view()->horizontalScrollBar()->value();
			_lines_view->set_all_zones_width([=](uint32_t width){ return width+zoom; });
			update_zones_position();
			/*launch_job_future([&](PVRenderingJob& rendering_job)
				{
					return _lines_view->render_all(view_x, view()->width(), rendering_job);
				}
			);*/
		}
		event->accept();
	}

	void translate_and_update_zones_position()
	{
		cancel_current_job();
		uint32_t view_x = view()->horizontalScrollBar()->value();
		uint32_t view_width = view()->width();
		//_lines_view->translate(view_x, view()->width());
		//update_zones_position();
		launch_job_future([&, view_x, view_width](PVRenderingJob& rendering_job)
			{
				return _lines_view->translate(view_x, view_width, rendering_job);
			}
		);
	}

private slots:
	void update_zone_pixmap(int zid)
	{
		if (!_lines_view->is_zone_drawn(zid)) {
			return;
		}
		PVParallelView::PVLinesView::list_zone_images_t& images = _lines_view->get_zones_images();
		const PVZoneID img_id = zid-_lines_view->get_first_drawn_zone();

		// Check whether the image needs scaling.
		QImage qimg = images[img_id].bg->qimage();
		QImage final_img;
		const uint32_t zone_width = _lines_view->get_zone_width(zid);
		if ((uint32_t) qimg.width() != zone_width) {
			final_img = qimg.scaled(zone_width, qimg.height(), Qt::IgnoreAspectRatio, Qt::FastTransformation);
		}
		else {
			final_img = qimg;
		}

		// Convert the image to a pixmap
		_zones[img_id]->setPixmap(QPixmap::fromImage(final_img));
		_zones[img_id]->setPos(QPointF(_lines_view->get_zone_absolute_pos(zid), 0));
	}

private:
	template <class F>
	void launch_job_future(F const& f)
	{
		// Launch our new job !
		_rendering_job->reset();
		_rendering_future = f(*_rendering_job);
	}

	void cancel_current_job()
	{
		if (_rendering_future.isRunning()) {
			PVLOG_INFO("(launch_job_future) Current job is running.. Cancelling it !\n");
			tbb::tick_count start = tbb::tick_count::now();
			// One job is running.. Ask the job to cancel and wait for it !
			_rendering_job->cancel();
			_rendering_future.waitForFinished();
			tbb::tick_count end = tbb::tick_count::now();
			PVLOG_INFO("(launch_job_future) Job ancellation done in %0.4f ms.\n", (end-start).seconds()*1000.0);
		}
	}

	void wait_end_current_job()
	{
		if (_rendering_future.isRunning()) {
			_rendering_future.waitForFinished();
		}
	}

	QRect map_to_axis(PVZoneID zid, QRectF rect)
	{
		QPointF point = _axes[zid]->map_from_scene(rect.topLeft());

		return QRect(point.x(), point.y(), rect.width(), rect.height());
	}

private slots:
	void slider_pressed_Slot()
	{
		_translation_start_x = (qreal) view()->horizontalScrollBar()->value();
	}

	void slider_released_Slot()
	{
		translate_and_update_zones_position();
	}

private:
    PVParallelView::PVLinesView* _lines_view;
    qreal _translation_start_x;

    QList<QGraphicsPixmapItem*> _zones;
    QList<PVParallelView::PVAxisWidget*> _axes;

	PVRenderingJob* _rendering_job;
	QFuture<void> _rendering_future;
    
	PVParallelView::PVSelectionSquareGraphicsItem* _selection_square;
    QPointF _selection_square_pos;
};

}

#endif // __PVPARALLELSCENE_h__
