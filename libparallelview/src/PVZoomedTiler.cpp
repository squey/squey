
#include <pvkernel/core/picviz_bench.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVZoomedTiler.h>

#include <pvparallelview/PVZonesDrawing.h>
#include <pvparallelview/PVBCIBackendImage_types.h>

#include <QGraphicsSceneMouseEvent>
#include <QGraphicsSceneWheelEvent>
#include <QGraphicsPixmapItem>
#include <QScrollBar>

// TODO: make it more C++ish
int Y_TILE_NUMBER = 3;
int Y_TILE_OFFSET = 3 * IMAGE_HEIGHT;
float UPDATE_SCALE = 1.3;


struct PVParallelView::PVZoomedTile
{
	PVParallelView::PVBCIBackendImage_p  bimage;
	QGraphicsPixmapItem *pixmap;

	/* The method ::set_visibility() is a work-around for a known bug in
	 * Qt which extends the scene rectangle to the rectangle normally used
	 * by the horizontal scrollbar. This bug is referenced QTBUG-14711:
	 * "QGraphicsView does not fully respect scrollbar policies set to
	 * Qt::ScrollBarAlwaysOff". for further information, see following
	 * URI: https://bugreports.qt-project.org/browse/QTBUG-14711
	 *
	 * When this bug will be corrected:
	 *  - remove ::set_visibility()
	 *  - move PVZoomedTiler::update_tile_position() to
	      PVZoomedTile::set_position()
	 */
	inline bool set_visibility(int zoom)
	{
		int limit = IMAGE_HEIGHT * (1 << zoom);

		if ((int)pixmap->pos().y() >= limit) {
			if (pixmap->isVisible() == true) {
				pixmap->setVisible(false);
			}
			return false;
		} else {
			if (pixmap->isVisible() == false) {
				pixmap->setVisible(true);
			}
			return true;
		}
	}
};





/*****************************************************************************/

/*****************************************************************************
 * PVParallelView::PVZoomedTiler::PVZoomedTiler()
 *****************************************************************************/

PVParallelView::PVZoomedTiler::PVZoomedTiler(QObject *parent, PVParallelView::PVZonesDrawing &zones_drawing,
                                             PVCol axis, uint32_t position, int zoom) :
	QGraphicsScene(parent),
	_zones_drawing(zones_drawing),
	_axis(axis),
	_zoom(zoom)
{
	QImage qi(1024, 1024, QImage::Format_RGB32);

	_left_tiles = _right_tiles = 0;

	if(axis > 0) {
		_left_tiles = new PVZoomedTile[Y_TILE_NUMBER];

		for (int i = 0; i < Y_TILE_NUMBER; ++i) {
			PVParallelView::PVBCIBackendImage_p img = zones_drawing.create_image(IMAGE_HEIGHT);
			_left_tiles[i].bimage = img;
			_left_tiles[i].pixmap = addPixmap(QPixmap::fromImage(img->qimage()));

			// strange way to set position, isn't it ? :-p
			_left_tiles[i].pixmap->setX(-IMAGE_HEIGHT);
			update_tile_position(_left_tiles[i], i * IMAGE_HEIGHT);
		}
	}

	if (axis < zones_drawing.get_zones_manager().get_number_cols()) {
		_right_tiles = new PVZoomedTile[Y_TILE_NUMBER];

		for (int i = 0; i < Y_TILE_NUMBER; ++i) {
			PVParallelView::PVBCIBackendImage_p img = zones_drawing.create_image(IMAGE_HEIGHT);
			_right_tiles[i].bimage = img;
			_right_tiles[i].pixmap = addPixmap(QPixmap::fromImage(img->qimage()));

			// same player shoots again
			_right_tiles[i].pixmap->setX(0);
			update_tile_position(_right_tiles[i], i * IMAGE_HEIGHT);
		}
	}

	view()->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	setBackgroundBrush(Qt::black);
	view()->verticalScrollBar()->setValue(position);

	// it will automatically invalidate all tiles to force their rendering
	update_scene_space();

	connect(view()->verticalScrollBar(), SIGNAL(sliderMoved(int)),
	        this, SLOT(update_tiles_Slot(int)));
	connect(view()->verticalScrollBar(), SIGNAL(actionTriggered(int)),
	        this, SLOT(update_tiles_Slot(int)));
	connect(view()->verticalScrollBar(), SIGNAL(sliderReleased()),
	        this, SLOT(update_tiles_Slot()));
}

/*****************************************************************************
 * PVParallelView::PVZoomedTiler::~PVZoomedTiler()
 *****************************************************************************/

PVParallelView::PVZoomedTiler::~PVZoomedTiler()
{
	if (_left_tiles) {
		delete [] _left_tiles;
	}
	if (_right_tiles) {
		delete [] _right_tiles;
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedTiler::mouseMoveEvent()
 *****************************************************************************/

void PVParallelView::PVZoomedTiler::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
	if (event->buttons() == Qt::RightButton) {
		QScrollBar *sb = view()->verticalScrollBar();
		int offset = _translation_start_y - event->scenePos().y();
		sb->setValue(sb->value() + offset);
		update_tiles_Slot();
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedTiler::mousePressEvent()
 *****************************************************************************/

void PVParallelView::PVZoomedTiler::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
	if (event->button() == Qt::RightButton) {
		_translation_start_y = event->scenePos().y();
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedTiler::mouseReleaseEvent()
 *****************************************************************************/

void PVParallelView::PVZoomedTiler::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
	// RH: is there a real thing to do?
	if (event->button() == Qt::RightButton) {
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedTiler::wheelEvent()
 *****************************************************************************/

void PVParallelView::PVZoomedTiler::wheelEvent(QGraphicsSceneWheelEvent* event)
{
	if (event->modifiers() == Qt::ControlModifier) {
		if (event->delta() > 0) {
			if (_zoom < 20) {
				++_zoom;
				update_scene_space(ZOOM_IN);
			}
		} else {
			if (_zoom > 0) {
				--_zoom;
				update_scene_space(ZOOM_OUT);
			}
		}
		event->accept();
	} else {
		update_tiles_Slot();
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedTiler::resizeEvent()
 *****************************************************************************/

void PVParallelView::PVZoomedTiler::resizeEvent(QResizeEvent* /*event*/)
{
	update_scene_space();
}

/*****************************************************************************
 * PVParallelView::PVZoomedTiler::update_scene_space()
 *****************************************************************************/

void PVParallelView::PVZoomedTiler::update_scene_space(pv_zoom_type_t zoom_type)
{
	QScrollBar *sb = view()->verticalScrollBar();
	int max_value = IMAGE_HEIGHT * (1 << _zoom);

	sb->setRange(0, max_value - IMAGE_HEIGHT);
	setSceneRect(-512, 0., 1024, max_value);

	// rescale the view to keep its aspect ratio
	view()->resetTransform();
	view()->scale(_zoom + 1, 1);

	if (zoom_type == ZOOM_IN) {
		sb->setValue(sb->value() << 1);
	} else if(zoom_type == ZOOM_OUT) {
		sb->setValue(sb->value() >> 1);
	}

	update_tiles_Slot();

	/* invalidating all tiles
	 */
	for (int i = 0; i < Y_TILE_NUMBER; ++i) {
		update_tiles_row(i);
		if (_left_tiles) {
			invalidate_tile(_left_tiles[i], true);
		}
		if (_right_tiles) {
			invalidate_tile(_right_tiles[i], false);
		}
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedTiler::update_tiles_row_position()
 *****************************************************************************/

void PVParallelView::PVZoomedTiler::update_tiles_row_position(int tile_num,
                                                              int new_y_pos)
{
	if (_left_tiles) {
		update_tile_position(_left_tiles[tile_num], new_y_pos);
		invalidate_tile(_left_tiles[tile_num], true);
	}
	if (_right_tiles) {
		update_tile_position(_right_tiles[tile_num], new_y_pos);
		invalidate_tile(_right_tiles[tile_num], true);
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedTiler::update_tiles_row()
 *****************************************************************************/

void PVParallelView::PVZoomedTiler::update_tiles_row(int tile_num)
{
	QScrollBar *sb = view()->verticalScrollBar();
	int pos = sb->value();
	QGraphicsPixmapItem *pixmap = ((_left_tiles)?_left_tiles:_right_tiles)[tile_num].pixmap;

	/* check for a move down
	 */
	int new_y_pos = pixmap->pos().y();
	int old_y_pos = new_y_pos;
	while ((pos - new_y_pos) > (IMAGE_HEIGHT * UPDATE_SCALE)) {
		new_y_pos += Y_TILE_OFFSET;
	}

	/* the tile has to be moved down
	 */
	if ((new_y_pos != old_y_pos) && (new_y_pos < sb->maximum())) {
		update_tiles_row_position(tile_num, new_y_pos);
	}

	/* check for a move up
	 */
	new_y_pos = pixmap->pos().y();
	old_y_pos = new_y_pos;
	while ((new_y_pos - pos) > (IMAGE_HEIGHT * UPDATE_SCALE)) {
		new_y_pos -= Y_TILE_OFFSET;
	}

	/* the tile has to be moved up
	 */
	if ((new_y_pos != old_y_pos) && (new_y_pos >= 0)) {
		update_tiles_row_position(tile_num, new_y_pos);
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedTiler::invalidate_tile()
 *****************************************************************************/

void PVParallelView::PVZoomedTiler::invalidate_tile(PVZoomedTile &tile, bool is_left)
{
	QGraphicsPixmapItem *pixmap = tile.pixmap;
	uint32_t y_min;

	if (tile.set_visibility(_zoom) == false) {
		return;
	}

	if (_zoom == 0) {
		y_min = 0;
	} else {
		y_min = ((int)pixmap->pos().y() / 1024) * (1 << (32 - _zoom));
	}

	// std::cout << "redrawing tile at " << y_min << std::endl;

	BENCH_START(render);
	if (is_left) {
		_zones_drawing.draw_zoomed_zone(*tile.bimage, y_min, _zoom, _axis - 1,
		                                &PVZoomedZoneTree::browse_tree_bci_by_y2);
	} else {
		_zones_drawing.draw_zoomed_zone(*tile.bimage, y_min, _zoom, _axis,
		                                &PVZoomedZoneTree::browse_tree_bci_by_y1);
	}
	BENCH_END(render, "render tile", 1, 1, 1, 1);
	pixmap->setPixmap(QPixmap::fromImage(tile.bimage->qimage()));
}

/*****************************************************************************
 * PVParallelView::PVZoomedTiler::update_tile_position()
 *****************************************************************************/

void PVParallelView::PVZoomedTiler::update_tile_position(PVZoomedTile &tile,
                                                         int new_y_pos)
{
	tile.pixmap -> setPos(QPointF (tile.pixmap->pos().x(), new_y_pos));
	tile.set_visibility(_zoom);
}

/*****************************************************************************
 * PVParallelView::PVZoomedTiler::update_tiles_Slot()
 *****************************************************************************/

void PVParallelView::PVZoomedTiler::update_tiles_Slot(int /*value*/)
{
	for (int i = 0; i < Y_TILE_NUMBER; ++i) {
		update_tiles_row(i);
	}
}

