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
#include <picviz/PVView_types.h>

#include <pvparallelview/PVBCIBackendImage_types.h>
#include <pvparallelview/PVSelectionSquareGraphicsItem.h>
#include <pvparallelview/PVSelectionGenerator.h>
#include <pvparallelview/PVAxisGraphicsItem.h>
#include <pvparallelview/PVFullParallelView.h>
#include <pvparallelview/PVLinesView.h>
#include <pvparallelview/PVSlidersManager.h>

#include <pvhive/PVActor.h>
#include <pvhive/PVCallHelper.h>
#include <pvhive/PVFuncObserver.h>

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
	typedef PVSlidersManager::axis_id_t axis_id_t;

public:
	PVFullParallelScene(PVFullParallelView* full_parallel_view, Picviz::PVView_sp& view_sp, PVParallelView::PVSlidersManager_p sm_p, PVBCIDrawingBackend& backend, PVZonesManager const& zm, PVZonesProcessor& zp_sel, PVZonesProcessor& zp_bg);
	virtual ~PVFullParallelScene();

	void first_render();
	void update_all_with_timer();

	void update_viewport();
	void update_scene(QGraphicsSceneWheelEvent* event = nullptr);

	void about_to_be_deleted();

	PVFullParallelView* graphics_view() { return _full_parallel_view; }

	void set_enabled(bool value)
	{
		if (!value) {
			_lines_view.cancel_and_wait_all_rendering();
		}
		_full_parallel_view->setEnabled(value);
	}

	void update_new_selection_async();
	void update_all_async();
	void update_number_of_zones_async();

private slots:
	void update_new_selection();
	void update_all();
	void update_number_of_zones();

private:
	void update_zones_position(bool update_all = true, bool scale = true);
	void translate_and_update_zones_position();

	void store_selection_square();
	void update_selection_square();

	void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
	void mouseMoveEvent(QGraphicsSceneMouseEvent* event) override;
	void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;
	void wheelEvent(QGraphicsSceneWheelEvent* event) override;
	void keyPressEvent(QKeyEvent* event) override;

	inline QPointF map_to_axis(PVZoneID zone_id, QPointF p) const { return _axes[zone_id]->mapFromScene(p); }
	inline QPointF map_from_axis(PVZoneID zone_id, QPointF p) const { return _axes[zone_id]->mapToScene(p); }
	QRect map_to_axis(PVZoneID zone_id, QRectF rect) const
	{
		QRect r = _axes[zone_id]->map_from_scene(rect);

		// top and bottom must be corrected according to the y zoom factor
		r.setTop(r.top() / _zoom_y);
		r.setBottom(r.bottom() / _zoom_y);

		const int32_t zone_width = _lines_view.get_zone_width(zone_id);
		if (r.width() + r.x() > zone_width) {
			r.setRight(zone_width-1);
		}

		return r;
	}

	bool sliders_moving() const;

	void process_selection();

	Picviz::PVView& lib_view() { return _lib_view; }
	Picviz::PVView const& lib_view() const { return _lib_view; }

	void add_zone_image();
	void add_axis(PVZoneID const zone_id, int index = -1);

	inline PVBCIDrawingBackend& backend() const { return _lines_view.backend(); }

private slots:
	void update_zone_pixmap_bg(int zone_id);
	void update_zone_pixmap_sel(int zone_id);
	void update_zone_pixmap_bgsel(int zone_id);
	void scale_zone_images(PVZoneID zone_id);

	void update_selection_from_sliders_Slot(axis_id_t axis_id);
	void scrollbar_pressed_Slot();
	void scrollbar_released_Slot();
	void commit_volatile_selection_Slot();

	void clear_selection_square()
	{
		_selection_barycenter.clear();
		_selection_square->clear_rect();
	}
	void emit_new_zoomed_parallel_view(int axis_index)
	{
		emit _full_parallel_view->new_zoomed_parallel_view(&_lib_view, axis_index);
	}

private slots:
	// Slots called from PVLinesView
	void zr_sel_finished(void* zr, int zone_id);
	void zr_bg_finished(void* zr, int zone_id);

	void render_all_zones_all_imgs();

private:
	int32_t pos_last_axis() const;

private:
	struct ZoneImages
	{
		QGraphicsPixmapItem* sel;
		QGraphicsPixmapItem* bg;

		PVBCIBackendImage_p img_tmp_sel;
		PVBCIBackendImage_p img_tmp_bg;

		ZoneImages()
		{
		}

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

		void hide()
		{
			sel->hide();
			bg->hide();
		}

		void show()
		{
			sel->show();
			bg->show();
		}

		void remove(QGraphicsScene* scene)
		{
			scene->removeItem(sel);
			scene->removeItem(bg);
		}
	};

	struct SelectionBarycenter
	{
		SelectionBarycenter()
		{
			clear();
		}

		PVZoneID zone_id1;
		PVZoneID zone_id2;
		double factor1;
		double factor2;

		void clear()
		{
			zone_id1 = PVZONEID_INVALID;
			zone_id2 = PVZONEID_INVALID;
			factor1 = 0.0;
			factor2 = 0.0;
		}
	};

private:
	typedef std::vector<PVParallelView::PVAxisGraphicsItem*> axes_list_t;

private:
	PVParallelView::PVLinesView _lines_view;

	std::vector<ZoneImages> _zones;
	axes_list_t             _axes;

	PVHive::PVActor<Picviz::PVView> _view_actor;
	Picviz::PVView& _lib_view;

	PVFullParallelView* _full_parallel_view;

	PVSelectionSquareGraphicsItem* _selection_square;
	SelectionBarycenter _selection_barycenter;
	PVParallelView::PVSelectionGenerator _selection_generator;
	QPointF _selection_square_pos;
	qreal _translation_start_x = 0.0;

	float           _zoom_y;
	float           _axis_length;

	PVSlidersManager_p _sm_p;

	QTimer* _timer_render;
};

}

#endif // __PVFULLPARALLELSCENE_h__
