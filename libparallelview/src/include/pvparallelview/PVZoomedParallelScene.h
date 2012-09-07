/**
 * \file PVZoomedParallelScene.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_PVZOOMEDPARALLELSCENE_H
#define PVPARALLELVIEW_PVZOOMEDPARALLELSCENE_H

#include <pvparallelview/common.h>
#include <pvparallelview/PVZoomedParallelView.h>
#include <pvparallelview/PVZonesDrawing.h>
#include <pvparallelview/PVRenderingJob.h>
#include <pvparallelview/PVSelectionSquareGraphicsItem.h>

#include <picviz/FakePVView.h>

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

public:
	PVZoomedParallelScene(PVParallelView::PVZoomedParallelView *zpview,
	                      Picviz::FakePVView_p pvview_p,
	                      zones_drawing_t &zones_drawing,
	                      PVCol axis);

	~PVZoomedParallelScene();

	void mousePressEvent(QGraphicsSceneMouseEvent *event);
	void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
	void mouseMoveEvent(QGraphicsSceneMouseEvent *event);

	void wheelEvent(QGraphicsSceneWheelEvent* event);

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

	PVZoomedParallelView          *_zpview;
	Picviz::FakePVView_p           _pvview_p;
	zones_drawing_t               &_zones_drawing;
	Picviz::PVSelection           &_selection;
	PVCol                          _axis;

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

	// about selection in the zoom view
	QPointF                        _selection_rect_pos;
	PVSelectionSquareGraphicsItem *_selection_rect;

	// about rendering invalidation
	render_t                       _render_type;
	int                            _rendering_zone_number;
	int                            _rendered_zone_count;
};

}

#endif // PVPARALLELVIEW_PVZOOMEDPARALLELSCENE_H
