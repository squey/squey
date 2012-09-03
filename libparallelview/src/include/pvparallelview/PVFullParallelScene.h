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

#include <pvhive/PVCallHelper.h>
#include <pvhive/PVFuncObserver.h>

#include <picviz/FakePVView.h>

#include <tbb/task_group.h>

namespace tbb {
class task;
}

namespace PVParallelView {

class draw_zone_Observer: public PVHive::PVFuncObserverSignal<typename PVLinesView::zones_drawing_t, FUNC(PVLinesView::zones_drawing_t::draw_zone<decltype(&PVParallelView::PVZoneTree::browse_tree_bci)>)>
{
public:
	draw_zone_Observer(PVFullParallelScene* parent) : _parent(parent) {}
protected:
	virtual void update(arguments_deep_copy_type const& args) const;
private:
	PVFullParallelScene* _parent;
};

class draw_zone_sel_Observer: public PVHive::PVFuncObserverSignal<typename PVLinesView::zones_drawing_t, FUNC(PVLinesView::zones_drawing_t::draw_zone<decltype(&PVParallelView::PVZoneTree::browse_tree_bci_sel)>)>
{
public:
	draw_zone_sel_Observer(PVFullParallelScene* parent) : _parent(parent) {}
protected:
	virtual void update(arguments_deep_copy_type const& args) const;
private:
	PVFullParallelScene* _parent;
};

class PVFullParallelScene : public QGraphicsScene
{
	Q_OBJECT

	friend class draw_zone_Observer;
	friend class draw_zone_sel_Observer;
	friend class process_selection_Observer;
public:
	PVFullParallelScene(Picviz::FakePVView::shared_pointer view_sp, PVParallelView::PVZonesManager& zm, PVParallelView::PVLinesView::zones_drawing_t::bci_backend_t& bci_backend);
	virtual ~PVFullParallelScene();

	void first_render();
	void update_new_selection(tbb::task* root);

private:
	void update_zones_position(bool update_all = true);
	void translate_and_update_zones_position();


	void store_selection_square();
	void update_selection_square();

	void mousePressEvent(QGraphicsSceneMouseEvent* event);
	void mouseMoveEvent(QGraphicsSceneMouseEvent* event);
	void mouseReleaseEvent(QGraphicsSceneMouseEvent* event);
	void wheelEvent(QGraphicsSceneWheelEvent* event);

	template <class F>
	void launch_job_future(F const& f)
	{
		// Launch our new job !
		_rendering_job->reset();
		_rendering_future = f(*_rendering_job);
	}
	void cancel_current_job();
	void wait_end_current_job();

	inline QPointF map_to_axis(PVZoneID zid, QPointF p) const { return _axes[zid]->mapFromScene(p); }
	inline QPointF map_from_axis(PVZoneID zid, QPointF p) const { return _axes[zid]->mapToScene(p); }
	QRect map_to_axis(PVZoneID zid, QRectF rect) const { return _axes[zid]->map_from_scene(rect); }

	bool sliders_moving() const;

	void process_selection();

	void connect_draw_zone();
	void connect_draw_zone_sel();

private slots:
	void update_zone_pixmap_Slot(int zid);
	void update_selection_from_sliders_Slot(PVZoneID zid);
	void scrollbar_pressed_Slot();
	void scrollbar_released_Slot();
	void commit_volatile_selection_Slot();
	void draw_zone_sel_Slot(int zid, bool changed);
	void draw_zone_Slot(int zid, bool changed);

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

	struct SelectionBarycenter
	{
		PVZoneID zid1;
		PVZoneID zid2;
		double factor1;
		double factor2;
	};

private:
    PVParallelView::PVLinesView _lines_view;

    QList<ZoneImages> _zones;
    QList<PVParallelView::PVAxisGraphicsItem*> _axes;

	PVRenderingJob* _rendering_job;
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

    draw_zone_Observer* _draw_zone_observer;
    draw_zone_sel_Observer* _draw_zone_sel_observer;

	tbb::task_group _render_tasks_sel;
	tbb::task_group _render_tasks_bg;
};

}

#endif // __PVFULLPARALLELSCENE_h__
