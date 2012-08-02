/**
 * \file PVZoomedTiler.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVAPRALLELVIEW_PVZOOMEDTILER_H
#define PVAPRALLELVIEW_PVZOOMEDTILER_H

#include <pvbase/types.h>

#include <QGraphicsView>

/* TODO:
 *  - Add PVAxisGraphicsItem update when moving/zooming
 *
 * IMPORTANT NOTE:
 *  - read PVZoomedTile's comment in PVZoomedTiler.cpp about a bug in Qt
 *
 * NOTE:
 *  - It could be better to have a tile tree to render tile at different resolutions.
 *    It depends on the time of the tile renderings.
 *    2012-06-07: It is fast enough (for 300M lines, it needs 170 ms per tile for the
 *    lowest zoom level, so it is fast enough:)
 *  - The line restriction helps rendering few lines but it can hides relevant lines
 *    from the view. So it is a bad idea according to me. A better (and more accuracy)
 *    way is to use a usual viewport area and select relevant trees in the forest to
 *    search in for crossing lines.
 *  - with a correct zoom, a fast slider movement shows black screen (but when the slider is
 *    released, the tiling is correctly update), I don't think it can be corrected.
 */

namespace PVParallelView
{

class PVZonesDrawing;

struct PVZoomedTile;

class PVZoomedTiler : public QGraphicsScene
{
	Q_OBJECT

public:
	enum pv_zoom_type_t {
		ZOOM_NONE = 0,
		ZOOM_IN,
		ZOOM_OUT
	};

	PVZoomedTiler(QObject *parent, PVParallelView::PVZonesDrawing &zones_drawing,
	              PVCol axis, uint32_t position, int zoom);

	~PVZoomedTiler();

	void mouseMoveEvent(QGraphicsSceneMouseEvent *event);
	void mousePressEvent(QGraphicsSceneMouseEvent *event);
	void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
	void wheelEvent(QGraphicsSceneWheelEvent* event);

	void update_scene_space(pv_zoom_type_t zoom_type = ZOOM_NONE);

private:
	void update_tiles_row_position(int tile_num, int new_y_pos);
	void update_tiles_row(int tile_num);

	void invalidate_tile(PVZoomedTile &tile, bool is_left);
	void update_tile_position(PVZoomedTile &tile, int new_y_pos);

private:
	inline QGraphicsView* view()
	{
		return (QGraphicsView*) parent();
	}

private slots:
	void update_tiles_Slot(int value = 0);

private:
	PVZonesDrawing &_zones_drawing;
	PVCol           _axis;
	int             _zoom;
	PVZoomedTile   *_left_tiles;
	PVZoomedTile   *_right_tiles;
	int             _translation_start_y;
};

}

#endif // PVAPRALLELVIEW_PVZOOMEDTILER_H
