/**
 * \file PVZoomedParallelScene.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_PVZOOMEDPARALLELSCENE_H
#define PVPARALLELVIEW_PVZOOMEDPARALLELSCENE_H

#include <picviz/PVView.h>

#include <pvhive/PVHive.h>
#include <pvhive/PVFuncObserver.h>
#include <pvhive/PVCallHelper.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVZoomedParallelView.h>
#include <pvparallelview/PVZonesDrawing.h>
#include <pvparallelview/PVSlidersManager.h>
#include <pvparallelview/PVRenderingJob.h>
#include <pvparallelview/PVSelectionSquareGraphicsItem.h>

#include <QGraphicsPixmapItem>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsSceneWheelEvent>

#include <QObject>
#include <QDialog>
#include <QPaintEvent>
#include <QTimer>

namespace PVParallelView
{

class PVZoomedParallelScene : public QGraphicsScene
{
Q_OBJECT

private:
	friend class zoom_sliders_new_obs;

private:
	constexpr static size_t bbits = PARALLELVIEW_ZZT_BBITS;
	constexpr static uint32_t image_width = 512;
	constexpr static uint32_t image_height = 1024;
	constexpr static double bbits_alpha_scale = 1. / (1. + (bbits - 10));
	constexpr static int zoom_steps = 5;
	constexpr static double root_step = pow(2.0, 1.0 / zoom_steps);
	constexpr static int max_wheel_value = 21 * zoom_steps;
	constexpr static int axis_half_width = PARALLELVIEW_AXIS_WIDTH / 2;

private:
	typedef PVParallelView::PVZoomedZoneTree::context_t zzt_context_t;

public:
	typedef PVParallelView::PVZonesDrawing<bbits> zones_drawing_t;
	typedef typename zones_drawing_t::backend_image_p_t backend_image_p_t;

private:
	typedef typename zones_drawing_t::render_group_t render_group_t;

public:
	PVZoomedParallelScene(PVParallelView::PVZoomedParallelView *zpview,
	                      Picviz::PVView_sp& pvview,
	                      PVSlidersManager_p sliders_manager_p,
	                      zones_drawing_t &zones_drawing,
	                      PVCol axis);

	~PVZoomedParallelScene();

	void mousePressEvent(QGraphicsSceneMouseEvent *event);
	void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
	void mouseMoveEvent(QGraphicsSceneMouseEvent *event);

	void wheelEvent(QGraphicsSceneWheelEvent* event);

	void keyPressEvent(QKeyEvent *event);

	void invalidate_selection();
	void update_new_selection(tbb::task* root);

	virtual void drawBackground(QPainter *painter, const QRectF &rect);

	void resize_display();

private:
	inline void update_all()
	{
		_render_type = RENDER_ALL;
		update_display();
	}

	inline void update_sel()
	{
		_render_type = RENDER_SEL;
		update_display();
	}

	// must not be called directly, use ::update_all() or ::update_sel()
	void update_display();

	void update_zoom();

private:
	int get_zoom_level()
	{
		return _wheel_value / zoom_steps;
	}

	int get_zoom_step()
	{
		return _wheel_value % zoom_steps;
	}

	double get_scale_factor()
	{
		// Phillipe's magic formula: 2^n × a^k
		return pow(2, get_zoom_level()) * pow(root_step, get_zoom_step());
	}

	PVZonesManager& get_zones_manager() { return _zones_drawing.get_zones_manager(); }

	inline Picviz::PVSelection& volatile_selection() { return _pvview.get_volatile_selection(); }

private slots:
	void scrollbar_changed_Slot(int value);
	void scrollbar_timeout_Slot();
	void zone_rendered_Slot(int z);
	void filter_by_sel_finished_Slot(int zid, bool changed);
	void commit_volatile_selection_Slot();

private:
	typedef enum {
		RENDER_ALL,
		RENDER_SEL
	} render_t;

	struct zone_desc_t
	{
		backend_image_p_t    bg_image;   // the image for unselected/zomby lines
		backend_image_p_t    sel_image;  // the image for selected lines
		zzt_context_t        context;    // the extraction context for ZZT
		QGraphicsPixmapItem *item;       // the scene's element
		QPointF              next_pos;   // the item position of the next rendering
	};

	class zoom_sliders_new_obs :
		public PVHive::PVFuncObserverSignal<PVSlidersManager,
		                                    FUNC(PVSlidersManager::new_zoom_sliders)>
	{
	public:
		zoom_sliders_new_obs(PVZoomedParallelScene *parent) : _parent(parent)
		{}

		void update(arguments_deep_copy_type const& args) const
		{
			PVCol axis = std::get<0>(args);

			if (axis == _parent->_axis) {
				PVSlidersManager::id_t id = std::get<1>(args);

				if (id != _parent) {
					uint32_t y_min = std::get<2>(args);
					uint32_t y_max = std::get<3>(args);
					printf("##### add new zoom sliders: %d %p %u %u\n",
					       axis, id, y_min, y_max);
					// TODO: add sliders in _parent
				} else {
					// ignore my own add
				}
			}
		}

	private:
		PVZoomedParallelScene *_parent;
	};

private:

	PVZoomedParallelView          *_zpview;
	Picviz::PVView&                _pvview;
	PVSlidersManager_p             _sliders_manager_p;
	zones_drawing_t               &_zones_drawing;
	PVCol                          _axis;
	zoom_sliders_new_obs           _zsn_obs;

	// about mouse
	int                            _wheel_value;
	int                            _pan_reference_y;

	// about zones rendering/display
	zone_desc_t                   *_left_zone;
	zone_desc_t                   *_right_zone;
	qreal                          _next_beta;
	qreal                          _current_beta;

	// about rendering
	PVRenderingJob                *_rendering_job;
	QFuture<void>                  _rendering_future;
	QTimer                         _scroll_timer;
	render_group_t                 _render_group;

	// about selection in the zoom view
	QPointF                        _selection_rect_pos;
	PVSelectionSquareGraphicsItem *_selection_rect;

	// about rendering invalidation
	render_t                       _render_type;
	int                            _renderable_zone_number;
	int                            _updated_selection_count;
};

}

#endif // PVPARALLELVIEW_PVZOOMEDPARALLELSCENE_H
