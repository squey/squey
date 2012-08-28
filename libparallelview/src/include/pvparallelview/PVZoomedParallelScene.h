/**
 * \file PVZoomedParallelScene.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_PVZOOMEDPARALLELSCENE_H
#define PVPARALLELVIEW_PVZOOMEDPARALLELSCENE_H

#include <pvparallelview/common.h>
#include <pvparallelview/PVZonesDrawing.h>
#include <pvparallelview/PVRenderingJob.h>

#include <QGraphicsView>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsSceneWheelEvent>

#include <QObject>
#include <QDialog>
#include <QPaintEvent>

namespace PVParallelView
{

class PVZoomedParallelScene : public QGraphicsScene
{
Q_OBJECT

private:
	constexpr static size_t bbits = PARALLELVIEW_ZZT_BBITS;
	constexpr static uint32_t image_width = 512;
	constexpr static uint32_t image_height = PVParallelView::constants<bbits>::image_height;
	constexpr static double bbits_alpha_scale = 1. / (1. + (bbits - 10));
	constexpr static int zoom_steps = 5;
	constexpr static double root_step = pow(2.0, 1.0 / zoom_steps);
	constexpr static int max_wheel_value = 21 * zoom_steps;

public:
	typedef PVParallelView::PVZonesDrawing<bbits> zones_drawing_t;
	typedef typename zones_drawing_t::backend_image_p_t backend_image_p_t;

public:
	PVZoomedParallelScene(QWidget *parent,
	                      zones_drawing_t &zones_drawing,
	                      PVCol axis);

	void mousePressEvent(QGraphicsSceneMouseEvent *event);
	void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
	void mouseMoveEvent(QGraphicsSceneMouseEvent *event);

	void wheelEvent(QGraphicsSceneWheelEvent* event);

	virtual void drawBackground(QPainter *painter, const QRectF &rect);

	void update_display();
	void resize_display(const QSize &s);

private:
	void update_zoom(bool in = true);

private:
	inline QGraphicsView* view()
	{
		return (QGraphicsView*) parent();
	}

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
		return pow(2, _zoom_level) * pow(root_step, get_zoom_step());
	}

private slots:
	void scrollbar_changed_Slot(int value);
	void zone_rendered_Slot(int z);

private:
	struct zone_desc
	{
		QRect             area;
		QPoint            pos;
		backend_image_p_t image;
		QImage            back_image;
	};

	zones_drawing_t &_zones_drawing;
	PVCol            _axis;
	int              _wheel_value;
	int              _zoom_level;
	int              _old_sb_pos;
	zone_desc        _left_zone;
	zone_desc        _right_zone;
	PVRenderingJob  *_rendering_job;
	QFuture<void>    _rendering_future;
	bool             _skip_update_zoom;
};

}

#endif // PVPARALLELVIEW_PVZOOMEDPARALLELSCENE_H
