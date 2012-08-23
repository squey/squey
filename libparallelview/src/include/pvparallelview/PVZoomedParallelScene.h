/**
 * \file PVZoomedParallelScene.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_PVZOOMEDPARALLELSCENE_H
#define PVPARALLELVIEW_PVZOOMEDPARALLELSCENE_H

#include <pvparallelview/common.h>
#include <pvparallelview/PVZonesDrawing.h>

#include <QGraphicsView>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsSceneWheelEvent>

#include <QDialog>
#include <QPaintEvent>

namespace PVParallelView
{

	class PVZoomedParallelScene : public QGraphicsScene
	{
		constexpr static size_t bbits = PARALLELVIEW_ZZT_BBITS;
		constexpr static uint32_t image_width = 512;
		constexpr static uint32_t image_height = PVParallelView::constants<bbits>::image_height;
		constexpr static double bbits_alpha_scale = 1. / (1. + (bbits - 10));
		constexpr static int zoom_steps = 5;
		constexpr static double root_step = pow(2.0, 1.0 / zoom_steps);
		constexpr static int max_wheel_value = 20 * zoom_steps;

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

	private:
		void update_zoom()
		{
			_zoom_level = get_zoom_level();
			double s = get_scale_factor();

			view()->resetTransform();
			view()->scale(s, s);
			qreal ncy = view()->mapToScene(view()->viewport()->rect()).boundingRect().center().y();
			view()->centerOn(0., ncy);
		}

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

	private:
		zones_drawing_t  &_zones_drawing;
		PVCol             _axis;
		int               _wheel_value;
		int               _zoom_level;
		QImage            _back_image;
		backend_image_p_t _left_image;
		backend_image_p_t _right_image;
	};

}

#endif // PVPARALLELVIEW_PVZOOMEDPARALLELSCENE_H
