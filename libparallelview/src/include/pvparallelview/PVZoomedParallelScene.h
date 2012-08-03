/**
 * \file PVZoomedParallelScene.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_PVZOOMEDPARALLELSCENE_H
#define PVPARALLELVIEW_PVZOOMEDPARALLELSCENE_H

#include <pvbase/types.h>

#include <QGraphicsView>

#include <pvparallelview/common.h>
#include <pvparallelview/PVZonesDrawing.h>

namespace PVParallelView
{

class PVZoomedParallelScene : public QGraphicsScene
{
	constexpr static size_t bbits = NBITS_INDEX;
	constexpr static int zoom_steps = 5;
	constexpr static double root_step = pow(2.0, 1.0 / zoom_steps);
	constexpr static int tile_number = 3;

public:
	typedef PVParallelView::PVZonesDrawing<bbits> zones_drawing_t;
	typedef typename zones_drawing_t::backend_image_p_t backend_image_p_t;

private:
	struct zoomed_tile_t
	{
		backend_image_p_t bimage;
		QRectF            coord;
		int               number;
		bool              valid;
	};

public:
	/**
	 * CTOR
	 * @param parent the parent QObject
	 * @param zones_drawing the zone drawing object
	 * @param axis the axis we zoom on
	 * @param position
	 */
	PVZoomedParallelScene(QObject *parent,
	                      zones_drawing_t &zones_drawing,
	                      PVCol axis, uint32_t position, int zoom);

	~PVZoomedParallelScene();

	void mouseMoveEvent(QGraphicsSceneMouseEvent *event);
	void mousePressEvent(QGraphicsSceneMouseEvent *event);
	void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
	void wheelEvent(QGraphicsSceneWheelEvent* event);

	virtual void drawBackground(QPainter *painter, const QRectF &rect);

private:
	void draw_tile(QPainter *painter,
	               const QRectF &scene_rect,
	               const zoomed_tile_t &tile);

	void raster_tile_with_hinting(QImage &image,
	                              const QRectF &scene_rect,
	                              const zoomed_tile_t &tile);

private:
	inline QGraphicsView* view()
	{
		return (QGraphicsView*) parent();
	}

	void update_tile_position(int tile_index);
	void update_zoom();
	void invalidate_tiles();
	void check_tiles_validity();
	void render_tile(zoomed_tile_t &tile, bool is_left);

private:
	int get_tile_num()
	{
		// the true formula is: 2 * (1 << _zoom_level);;
		return 1 << (1 + _zoom_level);
	}

	int get_zoom_level()
	{
		return _wheel_value / zoom_steps;
	}

	int get_zoom_step()
	{
		return _wheel_value % zoom_steps;
	}

private:
	zones_drawing_t &_zones_drawing;
	PVCol            _axis;
	qreal            _translation_start_y;
	int              _wheel_value;
	int              _zoom_level;
	int              _old_zoom_level;
	QImage           _back_image;
	zoomed_tile_t   *_left_tiles;
	zoomed_tile_t   *_right_tiles;
};

}

#endif // PVPARALLELVIEW_PVZOOMEDPARALLELSCENE_H
