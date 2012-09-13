/**
 * \file PVFullParallelScene.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef __PVFULLPARALLELSCENE_h__
#define __PVFULLPARALLELSCENE_h__

#include <QGraphicsScene>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsSceneWheelEvent>

#include <picviz/PVAxis.h>
#include <pvparallelview/PVSelectionSquareGraphicsItem.h>
#include <pvparallelview/PVSelectionGenerator.h>
#include <pvparallelview/PVAxisGraphicsItem.h>
#include <pvparallelview/PVFullParallelView.h>
#include <pvparallelview/PVLinesView.h>
#include <pvparallelview/PVSlidersManager.h>

#include <pvhive/PVCallHelper.h>
#include <pvhive/PVFuncObserver.h>

#include <picviz/FakePVView.h>

#include <tbb/task_group.h>

#include <QFuture>

namespace tbb {
class task;
}

namespace PVParallelView {

class PVFullParallelScene : public QGraphicsScene
{
	Q_OBJECT

	friend class draw_zone_Observer;
	friend class draw_zone_sel_Observer;
	friend class process_selection_Observer;
public:
	PVFullParallelScene(Picviz::FakePVView::shared_pointer view_sp, PVParallelView::PVZonesManager& zm, PVParallelView::PVSlidersManager_p sm_p, PVParallelView::PVLinesView::zones_drawing_t::bci_backend_t& bci_backend, tbb::task* root_sel);
	virtual ~PVFullParallelScene();

	void first_render();
	void update_new_selection();

private:
	void update_zones_position(bool update_all = true);
	void translate_and_update_zones_position();


	void store_selection_square();
	void update_selection_square();

	void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
	void mouseMoveEvent(QGraphicsSceneMouseEvent* event) override;
	void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;
	void wheelEvent(QGraphicsSceneWheelEvent* event) override;
	void keyPressEvent(QKeyEvent* event) override;

	void cancel_current_job();
	void wait_end_current_job();

	inline QPointF map_to_axis(PVZoneID zid, QPointF p) const { return _axes[zid]->mapFromScene(p); }
	inline QPointF map_from_axis(PVZoneID zid, QPointF p) const { return _axes[zid]->mapToScene(p); }
	QRect map_to_axis(PVZoneID zid, QRectF rect) const { return _axes[zid]->map_from_scene(rect); }

	bool sliders_moving() const;

	void process_selection();

	void connect_draw_zone_sel();
	void connect_rendering_job();

private slots:
	void update_zone_pixmap_bg(int zid);
	void update_zone_pixmap_sel(int zid);
	void update_zone_pixmap_bgsel(int zid);
	void scale_zone_images(PVZoneID zid);

	void update_selection_from_sliders_Slot(PVZoneID zid);
	void scrollbar_pressed_Slot();
	void scrollbar_released_Slot();
	void commit_volatile_selection_Slot();
	void draw_zone_sel_Slot(int zid, bool changed);
	void try_to_launch_zoom_job();

private:
	struct ZoneImages
	{
		QGraphicsPixmapItem* sel;
		QGraphicsPixmapItem* bg;

		PVLinesView::backend_image_p_t img_tmp_sel;
		PVLinesView::backend_image_p_t img_tmp_bg;

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

	struct SelectionBarycenter
	{
		SelectionBarycenter():
			zid1(PVZONEID_INVALID), zid2(PVZONEID_INVALID)
		{ }
		PVZoneID zid1;
		PVZoneID zid2;
		double factor1;
		double factor2;
	};

private:
    PVParallelView::PVLinesView _lines_view;

	std::vector<ZoneImages> _zones;
    QList<PVParallelView::PVAxisGraphicsItem*> _axes;

	PVRenderingJob* _rendering_job_sel;
	PVRenderingJob* _rendering_job_bg;
	PVRenderingJob* _rendering_job_all;

	QFuture<void> _rendering_future;
	QFuture<void> _sel_rendering_future;
    
	Picviz::FakePVView::shared_pointer _view_sp;
	PVFullParallelView* _parallel_view;

	PVSelectionSquareGraphicsItem* _selection_square;
	SelectionBarycenter _selection_barycenter;
	PVParallelView::PVSelectionGenerator _selection_generator;
	Picviz::PVSelection& _sel;
    QPointF _selection_square_pos;
    qreal _translation_start_x = 0.0;

	tbb::task_group _render_tasks_sel;
	tbb::task_group _render_tasks_bg;

	tbb::task* _root_sel;

	QTimer* _heavy_job_timer;
	QFuture<void> _task_waiter;
};

}

#endif // __PVFULLPARALLELSCENE_h__
