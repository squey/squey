/**
 * \file PVZoomedParallelScene.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/picviz_bench.h>

#include <pvparallelview/PVZoomedParallelScene.h>

#include <QScrollBar>
#include <QGraphicsSceneMouseEvent>

#define SCENE_HEIGHT IMAGE_HEIGHT

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::PVZoomedParallelScene
 *****************************************************************************/

PVParallelView::PVZoomedParallelScene::PVZoomedParallelScene(QObject *parent,
                                                             zones_drawing_t &zones_drawing,
                                                             PVCol axis, uint32_t position,
                                                             int zoom) :
	QGraphicsScene(parent),
	_zones_drawing(zones_drawing), _axis(axis), _wheel_value(0),
	_left_tiles(nullptr), _right_tiles(nullptr)
{
	setBackgroundBrush(Qt::black);
	view()->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	view()->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
	view()->setResizeAnchor(QGraphicsView::AnchorViewCenter);
	view()->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
	view()->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);

	// TODO: use/init position, zoom, _old_zoom_level
	//view()->verticalScrollBar()->setValue(position);

	_wheel_value = 0;
	_old_zoom_level = get_zoom_level();

	if(axis > 0) {
		_left_tiles = new zoomed_tile_t[tile_number];

		for (int i = 0; i < tile_number; ++i) {
			backend_image_p_t img = zones_drawing.create_image(IMAGE_HEIGHT);
			_left_tiles[i].bimage = img;
		}
	}

	if (axis < zones_drawing.get_zones_manager().get_number_cols()) {
		_right_tiles = new zoomed_tile_t[tile_number];

		for (int i = 0; i < tile_number; ++i) {
			backend_image_p_t img = zones_drawing.create_image(IMAGE_HEIGHT);
			_right_tiles[i].bimage = img;
		}
	}

	_back_image = QImage(view()->width(), view()->height(), QImage::Format_ARGB32);

	update_zoom();

	// make sure all tiles are invalid
	invalidate_tiles();
	check_tiles_validity();
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::~PVZoomedParallelScene
 *****************************************************************************/

PVParallelView::PVZoomedParallelScene::~PVZoomedParallelScene()
{
	if (_left_tiles) {
		delete [] _left_tiles;
	}
	if (_right_tiles) {
		delete [] _right_tiles;
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::mouseMoveEvent
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
	if (event->buttons() == Qt::RightButton) {
		QScrollBar *sb = view()->verticalScrollBar();
		long offset = (long)(_translation_start_y - event->scenePos().y());
		sb->setValue(sb->value() + offset);
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::mousePressEvent
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
	if (event->button() == Qt::RightButton) {
		_translation_start_y = event->scenePos().y();
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::mouseReleaseEvent
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::mouseReleaseEvent(QGraphicsSceneMouseEvent */*event*/)
{
	// do nothing to avoid the behaviour of QGraphicsScene::mouseReleaseEvent()
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::wheelEvent
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::wheelEvent(QGraphicsSceneWheelEvent* event)
{
	if (event->modifiers() == Qt::ControlModifier) {
		if (event->delta() > 0) {
			if (_wheel_value < 100) {
				++_wheel_value;
				update_zoom();
				check_tiles_validity();
			}
		} else {
			if (_wheel_value > 0) {
				--_wheel_value;
				update_zoom();
				check_tiles_validity();
			}
		}
	} else if (event->modifiers() == Qt::NoModifier) {
		// default vertical translation
		QScrollBar *sb = view()->verticalScrollBar();
		if (event->delta() > 0) {
			sb->triggerAction(QAbstractSlider::SliderSingleStepSub);
		} else {
			sb->triggerAction(QAbstractSlider::SliderSingleStepAdd);
		}
	}

	event->accept();
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::drawBackground
 *****************************************************************************/

/* TODO: try to have a working one with QGraphicsView::ViewportUpdateMode ==
 *       QGraphicsView::MinimalViewportUpdate (which is the default value).
 *       I (RH) changed for QGraphicsView::FullViewportUpdate because with
 *       fast panning, the background was badly updated. A 1920x1280 view
 *       needs ~40 ms to update in the worst case (which is no so bad)
 */
void PVParallelView::PVZoomedParallelScene::drawBackground(QPainter *painter,
                                                           const QRectF &/*rect*/)
{
	QRect view_rect = view()->viewport()->rect();
	int view_center = view_rect.width() / 2;
	long tile_num = get_tile_num();

	_back_image = QImage(view_rect.size(), QImage::Format_ARGB32);
	_back_image.fill(Qt::black);

	// we need to save the painter's state
	QPen old_pen = painter->pen();
	QTransform t = painter->transform();
	painter->resetTransform();

	// and use our own painter for the QImage
	QPainter image_painter(&_back_image);

	// the tile set must be updated before any computation
	for (int i = 0; i < tile_number; ++i) {
		update_tile_position(i);
	}
	check_tiles_validity();

	BENCH_START(drawBackground);
	if (_left_tiles) {
		int decal = view_center - PARALLELVIEW_AXIS_WIDTH / 2;
		QRect left_rect(0, 0,
		                decal, view_rect.height());
		QRectF left_scene_rect = view()->mapToScene(left_rect).boundingRect();

		for (int i = 0; i < tile_number; ++i) {
			if (_left_tiles[i].number < tile_num) {
				draw_tile(&image_painter, left_scene_rect, _left_tiles[i]);
				// raster_tile_with_hinting(_back_image, left_scene_rect, _left_tiles[i]);
			}
		}
	}

	if (_right_tiles) {
		// view center + line width + half axis width
		int decal = view_center + 1 + PARALLELVIEW_AXIS_WIDTH / 2;
		QRect right_rect(decal, 0,
		                view_rect.width() - decal, view_rect.height());
		QRectF right_scene_rect = view()->mapToScene(right_rect).boundingRect();

		for (int i = 0; i < tile_number; ++i) {
			if (_right_tiles[i].number < tile_num) {
				draw_tile(&image_painter,right_scene_rect, _right_tiles[i]);
				// raster_tile_with_hinting(_back_image, right_scene_rect, _right_tiles[i]);
			}
		}
	}

	// blit to screen
	painter->drawImage(QPoint(0,0), _back_image);
	BENCH_END(drawBackground, "drawBackground", 1, 1, 1, 1);


	// draw axis
	QPen new_pen = QPen(Qt::white);
	new_pen.setColor(QColor(0xFFFFFFFF));
	new_pen.setWidth(PARALLELVIEW_AXIS_WIDTH);
	painter->setPen(new_pen);
	painter->drawLine(view_center, 0, view_center, view_rect.height());

	// get back the painter's original state
	painter->setTransform(t);
	painter->setPen(old_pen);
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::draw_tile
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::draw_tile(QPainter *painter,
                                                      const QRectF &scene_rect,
                                                      const zoomed_tile_t &tile)
{
	if(scene_rect.intersects(tile.coord) == false) {
		return;
	}

	QRectF inter = scene_rect.intersected(tile.coord);

	QRect screen_area = view()->mapFromScene(inter).boundingRect();

	// we need the sub-area of tile's image to draw
	QRect tile_rel;
	tile_rel.setLeft((int)(IMAGE_HEIGHT * (inter.x() - tile.coord.x()) / tile.coord.width()));
	tile_rel.setTop((int)(IMAGE_HEIGHT * (inter.y() - tile.coord.y()) / tile.coord.height()));

	tile_rel.setWidth((int)(IMAGE_HEIGHT * (inter.width() / tile.coord.width())));
	tile_rel.setHeight((int)(IMAGE_HEIGHT * (inter.height() / tile.coord.height())));

	painter->drawImage(screen_area, tile.bimage->qimage(), tile_rel);
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::raster_tile_with_hinting
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::raster_tile_with_hinting(QImage &image,
                                                                     const QRectF &scene_rect,
                                                                     const zoomed_tile_t &tile)
{
	if(scene_rect.intersects(tile.coord) == false) {
		return;
	}

	QRectF inter = scene_rect.intersected(tile.coord);

	QRect screen_area = view()->mapFromScene(inter).boundingRect();

	// we need the sub-area of tile's image to draw
	QRect tile_rel;
	tile_rel.setLeft((int)(IMAGE_HEIGHT * (inter.x() - tile.coord.x()) / tile.coord.width()));
	tile_rel.setTop((int)(IMAGE_HEIGHT * (inter.y() - tile.coord.y()) / tile.coord.height()));

	tile_rel.setWidth((int)(IMAGE_HEIGHT * (inter.width() / tile.coord.width())));
	tile_rel.setHeight((int)(IMAGE_HEIGHT * (inter.height() / tile.coord.height())));

	// TODO: replace the Code (tm) (c) :)
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::update_tile_position
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::update_tile_position(int tile_index)
{
	/* some init
	 */
	long tile_num = get_tile_num();
	double tile_scale = 1. / (double)tile_num;
	double tile_height = SCENE_HEIGHT / (double)tile_num;

	/* get the position of the upper tile (in tile's space). It is
	 * deduced from the view's top in scene's space
	 */
	long tile_upper_pos = (long)(view()->mapToScene(0, 0).y() / tile_height);

	/* if the processed tile has a lower index that the upper one, it
	 * must be shifted downward
	 */
	long tile_shift = tile_index - (tile_upper_pos % tile_number);
	if (tile_shift < 0) {
		tile_shift += tile_number;
	}

	double old_y;
	int number = tile_upper_pos + tile_shift;
	double new_y = (number) * tile_height;

	QSizeF tile_size(IMAGE_HEIGHT * .5, IMAGE_HEIGHT * tile_scale);

	// reconfigure all tiles
	if (_left_tiles) {
		zoomed_tile_t &tile = _left_tiles[tile_index];
		old_y = tile.coord.y();

		tile.coord.setTopLeft(QPointF(-512, new_y));
		tile.coord.setSize(tile_size);
		tile.number = number;

		tile.valid = (old_y == new_y);
	}

	if (_right_tiles) {
		zoomed_tile_t &tile = _right_tiles[tile_index];
		old_y = tile.coord.y();

		tile.coord.setTopLeft(QPointF(0, new_y));
		tile.coord.setSize(tile_size);
		tile.number = number;

		tile.valid = (old_y == new_y);
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::update_zoom
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::update_zoom()
{
	_zoom_level = get_zoom_level();
	int step = get_zoom_step();

	setSceneRect(-SCENE_HEIGHT * 0.5, 0., SCENE_HEIGHT, SCENE_HEIGHT);

	// Phillipe's magic formula: 2^n × a^k
	double s = pow(2, _zoom_level) * pow(root_step, step);

	view()->resetTransform();
	view()->scale(s, s);

	qreal ncy = view()->mapToScene(view()->viewport()->rect()).boundingRect().center().y();
	view()->centerOn(0., ncy);

	/* tiles must be against the axis line, they also must be shifted of
	 * pixel's width (in scene coordinates system) or 2 pixels width (in
	 * case of the right tiles)
	 */
	QRect screen_pixel_size(0, 0, 1, 0);
	QRectF scene_pixel_size = view()->mapToScene(screen_pixel_size).boundingRect();

	if (_left_tiles) {
		for (int i = 0; i < tile_number; ++i) {
			_left_tiles[i].coord.adjust(-scene_pixel_size.width(), 0, 0, 0);
		}
	}
	if (_right_tiles) {
		for (int i = 0; i < tile_number; ++i) {
			_left_tiles[i].coord.adjust(2 * scene_pixel_size.width(), 0, 0, 0);
		}
	}

	/* finally, if the zoom level has changed, all tiles must be
	 * invalidated to be recalculted
	 */
	if (_zoom_level != _old_zoom_level) {
		_old_zoom_level = _zoom_level;
		invalidate_tiles();
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::invalidate_tiles
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::invalidate_tiles()
{
	if (_left_tiles) {
		for (int i = 0; i < tile_number; ++i) {
			_left_tiles[i].valid = false;
		}
	}

	if (_right_tiles) {
		for (int i = 0; i < tile_number; ++i) {
			_right_tiles[i].valid = false;
		}
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::check_tiles_validity
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::check_tiles_validity()
{
	if (_left_tiles) {
		for (int i = 0; i < tile_number; ++i) {
			zoomed_tile_t &tile = _left_tiles[i];
			if (tile.valid == false) {
				render_tile(tile, true);
			}
		}
	}

	if (_right_tiles) {
		for (int i = 0; i < tile_number; ++i) {
			zoomed_tile_t &tile = _right_tiles[i];
			if (tile.valid == false) {
				render_tile(tile, false);
			}
		}
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::render_tile
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::render_tile(zoomed_tile_t &tile, bool is_left)
{
	long tile_num = get_tile_num();
	double tile_height = SCENE_HEIGHT / (double)tile_num;

	// we render at _zoom_level+1
	int level = _zoom_level + 1;

	uint32_t y_min = ((long)tile.coord.top() / tile_height) * (1 << (32 - level));

	BENCH_START(render);
	if (is_left) {
		_zones_drawing.draw_zoomed_zone(*tile.bimage, y_min, level, _axis - 1,
		                                &PVParallelView::PVZoomedZoneTree::browse_tree_bci_by_y2);
	} else {
		_zones_drawing.draw_zoomed_zone(*tile.bimage, y_min, level, _axis,
		                                &PVParallelView::PVZoomedZoneTree::browse_tree_bci_by_y1);
	}
	BENCH_END(render, "render tile", 1, 1, 1, 1);
}
