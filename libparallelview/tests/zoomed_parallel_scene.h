/**
 * \file zoomed_parallel_scene.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVZOOMEDPARALLELSCENE_H
#define PVZOOMEDPARALLELSCENE_H

#include <pvbase/types.h>

#include <QObject>
#include <QGraphicsView>
#include <QGraphicsLineItem>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsSceneWheelEvent>

namespace PVParallelView
{

class PVZonesDrawing;


}

struct PVZoomedTile;

class PVZoomedParallelScene : public QGraphicsScene
{
	Q_OBJECT

public:
	/**
	 * CTOR
	 * @param parent the parent QObject
	 * @param zones_drawing the zone drawing oject
	 * @param axis the axis we zoom on
	 * @param position
	 */
	PVZoomedParallelScene(QObject *parent,
	                      PVParallelView::PVZonesDrawing &zones_drawing,
	                      PVCol axis, uint32_t position, int zoom);

	~PVZoomedParallelScene();

	void mouseMoveEvent(QGraphicsSceneMouseEvent *event);
	void mousePressEvent(QGraphicsSceneMouseEvent *event);
	void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
	void wheelEvent(QGraphicsSceneWheelEvent* event);

private:
	inline QGraphicsView* view()
	{
		return (QGraphicsView*) parent();
	}

	void update_tile_position(int tile_index);
	void update_zoom();
	void invalidate_tiles();
	void check_tiles_validity();
	void render_tile(PVZoomedTile &tile, bool is_left);

	void update_axis_line();

private slots:
	void update_view(int value = 0);

private:
	PVParallelView::PVZonesDrawing &_zones_drawing;
	PVCol                           _axis;
	qreal                           _translation_start_y;
	int                             _wheel_value;
	int                             _old_level;
	QGraphicsLineItem              *_axis_line;
	PVZoomedTile                   *_left_tiles;
	PVZoomedTile                   *_right_tiles;
};

#endif // PVZOOMEDPARALLELSCENE_H
