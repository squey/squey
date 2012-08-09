/**
 * \file PVParallelScene.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef __PVPARALLELSCENE_h__
#define __PVPARALLELSCENE_h__

#include <QGraphicsScene>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsSceneWheelEvent>

#include <picviz/PVAxis.h>
#include <pvparallelview/PVSelectionSquareGraphicsItem.h>
#include <pvparallelview/PVSelectionGenerator.h>
#include <pvparallelview/PVAxisGraphicsItem.h>
#include <pvparallelview/PVFullParallelView.h>
#include <pvparallelview/PVLinesView.h>

#include <tbb/tick_count.h>

#define CRAND() (127 + (random() & 0x7F))

namespace PVParallelView {

class PVParallelScene : public QGraphicsScene
{
	Q_OBJECT

public:
	PVParallelScene(QObject* parent, PVParallelView::PVLinesView* lines_view);

	virtual ~PVParallelScene()
	{
		_rendering_job->deleteLater();
	}

	inline PVLinesView* get_lines_view() { return _lines_view; }

	void first_render()
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

	bool sliders_moving() const
	{
		for (PVAxisGraphicsItem* axis : _axes) {
			if (axis->sliders_moving()) {
				return true;
			}
		}
		return false;
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
		QGraphicsScene::mousePressEvent(event);
	}

	void mouseMoveEvent(QGraphicsSceneMouseEvent *event)
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
			_selection_square->setRect(QRectF(top_left, bottom_right));
		}
		QGraphicsScene::mouseMoveEvent(event);
	}

	void mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
	{
		if (event->button() == Qt::RightButton) {
			// translate zones
			translate_and_update_zones_position();
		}
		else if (!sliders_moving()) {
			PVZoneID zid = _lines_view->get_zones_manager().get_zone_id(_selection_square->rect().x());
			QRect r = map_to_axis(zid, _selection_square->rect());

			if (_selection_square_pos == event->scenePos()) {
				// Remove selection
				r = QRect(0, 0, 0, 0);
				_selection_square->setRect(r);
			}

			cancel_current_job();
			uint32_t nb_select = _selection_generator.compute_selection_from_rect(zid, r, _sel);
			view()->set_selected_line_number(nb_select);
			launch_job_future([&](PVRenderingJob& rendering_job)
				{
					return _lines_view->update_sel_from_zone(view()->width(), zid, _sel, rendering_job);
				}
			);
		}
		QGraphicsScene::mouseReleaseEvent(event);
	}

	void wheelEvent(QGraphicsSceneWheelEvent* event);

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


	void update_selection(uint32_t axis_id)
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
			PVLOG_INFO("(launch_job_future) Job cancellation done in %0.4f ms.\n", (end-start).seconds()*1000.0);
		}
	}

	void wait_end_current_job()
	{
		if (_rendering_future.isRunning()) {
			_rendering_future.waitForFinished();
		}
	}

	inline QPointF map_to_axis(PVZoneID zid, QPointF p) const { return _axes[zid]->mapFromScene(p); }
	inline QPointF map_from_axis(PVZoneID zid, QPointF p) const { return _axes[zid]->mapToScene(p); }
	QRect map_to_axis(PVZoneID zid, QRectF rect) const { return _axes[zid]->map_from_scene(rect); }

private slots:
	void scrollbar_pressed_Slot()
	{
		_translation_start_x = (qreal) view()->horizontalScrollBar()->value();
	}

	void scrollbar_released_Slot()
	{
		translate_and_update_zones_position();
	}

private:

	struct ZoneImages
	{
		QGraphicsPixmapItem* sel;
		QGraphicsPixmapItem* bg;

		void setPos(QPointF point)
		{
			sel->setPos(point);
			bg->setPos(point);
		}

		void setPixmap(QPixmap const& pixmap_sel, QPixmap const& pixmap_bg)
		{
			sel->setPixmap(pixmap_sel);
			bg->setPixmap(pixmap_bg);
		}
	};

    PVParallelView::PVLinesView* _lines_view;
    qreal _translation_start_x;

    QList<ZoneImages> _zones;
    QList<PVParallelView::PVAxisGraphicsItem*> _axes;

	PVRenderingJob* _rendering_job;
	QFuture<void> _rendering_future;
	QFuture<void> _sel_rendering_future;
    
	PVSelectionSquareGraphicsItem* _selection_square;
	PVParallelView::PVSelectionGenerator _selection_generator;

    QPointF _selection_square_pos;

    Picviz::PVSelection _sel;
};

}

#endif // __PVPARALLELSCENE_h__
