#ifndef PVAPRALLELVIEW_PVZOOMEDTILER_H
#define PVAPRALLELVIEW_PVZOOMEDTILER_H

#include <pvbase/types.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVZonesDrawing.h>

#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsPixmapItem>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsSceneWheelEvent>
#include <QScrollBar>

/* TODO:
 *  - make an invalidation mechanism (should not be too difficult:)
 *  - render the tile (*it will be* difficult:)
 *  - with a correct zoom, a fast slider movement shows black screen (but when the slider is
 *    released, the tiling is correctly update), I don't think I can correct that...
 *
 * NOTE:
 *  - It would be better to have a tile tree to render tile at different resolutions
 */

namespace PVParallelView
{

struct PVZoomedTile
{
	PVBCIBackendImage_p  bimage;
	QGraphicsPixmapItem *pixmap;
	bool                 is_valid;
};

namespace __impl {

// TODO: make it more C++ish
int Y_TILE_NUMBER = 3;
int Y_TILE_OFFSET = 3 * IMAGE_HEIGHT;

// TODO: make it more C++ish
enum pv_zoomed_tiler_zoom_t {
	NO_ZOOM,
	ZOOM_IN,
	ZOOM_OUT,
};

}

class PVZoomedTiler : public QGraphicsScene
{
	Q_OBJECT
public:
	PVZoomedTiler(QObject *parent, PVParallelView::PVZonesDrawing &zones_drawing,
	              PVCol axis,
	              uint32_t position, int zoom) :
		QGraphicsScene(parent),
		_zones_drawing(zones_drawing),
		_axis(axis),
		_zoom(zoom),
		_first_tile(0)
	{
		QImage qi(1024, 1024, QImage::Format_RGB32);

		_left_tiles = _right_tiles = 0;

		if(axis > 0) {
			_left_tiles = new PVZoomedTile [__impl::Y_TILE_NUMBER];

			for (int i = 0; i < __impl::Y_TILE_NUMBER; ++i) {
				PVBCIBackendImage_p img = zones_drawing.create_image(IMAGE_HEIGHT);
				_left_tiles[i].bimage = img;
				_left_tiles[i].pixmap = addPixmap(QPixmap::fromImage(img->qimage()));
				_left_tiles[i].pixmap->setPos(-IMAGE_HEIGHT, i * IMAGE_HEIGHT);

				// ugly way to update
				update_tile_position(_left_tiles[i], i * IMAGE_HEIGHT);
			}
		}

		if (axis < zones_drawing.get_zones_manager().get_number_cols()) {
			_right_tiles = new PVZoomedTile [__impl::Y_TILE_NUMBER];

			for (int i = 0; i < __impl::Y_TILE_NUMBER; ++i) {
				PVBCIBackendImage_p img = zones_drawing.create_image(IMAGE_HEIGHT);
				_right_tiles[i].bimage = img;
				_right_tiles[i].pixmap = addPixmap(QPixmap::fromImage(img->qimage()));
				_right_tiles[i].pixmap->setPos(0, i * IMAGE_HEIGHT);

				// ugly way to update
				update_tile_position(_right_tiles[i], i * IMAGE_HEIGHT);
			}
		}

		view()->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
		setBackgroundBrush(Qt::black);
		view()->verticalScrollBar()->setValue(position);

		// it will automatically invalidate all tiles to force their rendering
		update_scene_space();

		connect(view()->verticalScrollBar(), SIGNAL(sliderMoved(int)),
		        this, SLOT(update_tiling_Slot(int)));
		connect(view()->verticalScrollBar(), SIGNAL(actionTriggered(int)),
		        this, SLOT(update_tiling_Slot(int)));
		connect(view()->verticalScrollBar(), SIGNAL(sliderReleased()),
		        this, SLOT(update_tiling_Slot()));
	}

	~PVZoomedTiler()
	{
		if (_left_tiles) {
			delete _left_tiles;
		}
		if (_right_tiles) {
			delete _right_tiles;
		}
	}

	void mouseMoveEvent(QGraphicsSceneMouseEvent *event)
	{
		if (event->buttons() == Qt::RightButton) {
			QScrollBar *sb = view()->verticalScrollBar();
			int offset = _translation_start_y - event->scenePos().y();
			sb->setValue(sb->value() + offset);
			update_tiling_Slot();
		}
	}

	void mousePressEvent(QGraphicsSceneMouseEvent *event)
	{
		if (event->button() == Qt::RightButton) {
			_translation_start_y = event->scenePos().y();
		}
	}

	void mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
	{
		// RH: to finalize updates
		if (event->button() == Qt::RightButton) {
		}
	}

	void wheelEvent(QGraphicsSceneWheelEvent* event)
	{
		if (event->modifiers() == Qt::ControlModifier) {
			if (event->delta() > 0) {
				if (_zoom < 20) {
					++_zoom;
					update_scene_space(__impl::ZOOM_IN);
				}
			} else {
				if (_zoom > 0) {
					--_zoom;
					update_scene_space(__impl::ZOOM_OUT);
				}
			}
			event->accept();
		} else {
			update_tiling_Slot();
		}
	}

private:
	inline QGraphicsView* view()
	{
		return (QGraphicsView*) parent();
	}

	void update_scene_space(__impl::pv_zoomed_tiler_zoom_t zoom_type = __impl::NO_ZOOM)
	{
		QScrollBar *sb = view()->verticalScrollBar();
		int max_value = IMAGE_HEIGHT * (1 << _zoom);

		sb->setRange(0, max_value - IMAGE_HEIGHT);
		setSceneRect(-512., 0., 1024., max_value);

		if (zoom_type == __impl::ZOOM_IN) {
			sb->setValue(sb->value() << 1);
		} else if(zoom_type == __impl::ZOOM_OUT) {
			sb->setValue(sb->value() >> 1);
		}

		update_tiling_Slot();

		/* invalidating all tiles
		 */
		for (int i = 0; i < __impl::Y_TILE_NUMBER; ++i) {
			update_tiles_row(i);
			if (_left_tiles) {
				invalidate_tile(_left_tiles[i], true);
			}
			if (_right_tiles) {
				invalidate_tile(_right_tiles[i], false);
			}
		}
	}

	void update_tiles_row_position(int tile_num, int new_y_pos)
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

	void update_tiles_row(int tile_num)
	{
		QScrollBar *sb = view()->verticalScrollBar();
		int pos = sb->value();
		QGraphicsPixmapItem *pixmap = ((_left_tiles)?_left_tiles:_right_tiles)[tile_num].pixmap;

		/* check for a move down
		 */
		int new_y_pos = pixmap->pos().y();
		int old_y_pos = new_y_pos;
		while ((pos - new_y_pos) > (IMAGE_HEIGHT * 1.5)) {
			new_y_pos += __impl::Y_TILE_OFFSET;
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
		while ((new_y_pos - pos) > (IMAGE_HEIGHT * 1.5)) {
			new_y_pos -= __impl::Y_TILE_OFFSET;
		}

		/* the tile has to be moved up
		 */
		if ((new_y_pos != old_y_pos) && (new_y_pos >= 0)) {
			update_tiles_row_position(tile_num, new_y_pos);
		}
	}

private slots:
	void update_tiling_Slot(int /*value*/ = 0)
	{
		for (int i = 0; i < __impl::Y_TILE_NUMBER; ++i) {
			update_tiles_row(i);
		}
	}

	void update_slider_Slot()
	{
		std::cout << "PVZoomedTiler::update_slider_Slot()" << std::endl;
	}

private:
	/* The {is,set}Visible(true|false) code in:
	 *    ::update_tile_position(PVZoomedTile &tile, int new_y_pos)
	 * and
	 *    ::invalidate_tile(PVZoomedTile &tile, bool is_left)
	 *
	 * is a work-around for a known bug in Qt which extends the scene
	 * rectangle to the rectangle normally used by the horizontal
	 * scrollbar. This bug is referenced QTBUG-14711: "QGraphicsView does
	 * not fully respect scrollbar policies set to Qt::ScrollBarAlwaysOff".
	 * for further information, see following URI:
	 * https://bugreports.qt-project.org/browse/QTBUG-14711
	 *
	 * When this bug will be corrected, remove it.
	 */
	bool update_tile_visibility(PVZoomedTile &tile)
	{
		QGraphicsPixmapItem *pixmap = tile.pixmap;
		int limit = IMAGE_HEIGHT * (1 << _zoom);

		if ((int)pixmap->pos().y() >= limit) {
			if (pixmap->isVisible() == true) {
				pixmap->setVisible(false);
				std::cout << "hiding pixmap " << pixmap << std::endl;
			}
			return false;
		} else {
			if (pixmap->isVisible() == false) {
				pixmap->setVisible(true);
				std::cout << "showing pixmap " << pixmap << std::endl;
			}
			return true;
		}
	}

	inline void update_tile_position(PVZoomedTile &tile, int new_y_pos)
	{
		QGraphicsPixmapItem *pixmap = tile.pixmap;
		// std::cout << "new position for " << pixmap << ": " << new_y_pos << std::endl;
		// std::cout << "    limit: " << limit << std::endl;

		pixmap -> setPos(QPointF (pixmap->pos().x(), new_y_pos));
		update_tile_visibility(tile);
	}

	void invalidate_tile(PVZoomedTile &tile, bool is_left)
	{
		QGraphicsPixmapItem *pixmap = tile.pixmap;
		uint32_t y_min;

		if (update_tile_visibility(tile) == false) {
			return;
		}

		if (_zoom == 0) {
			y_min = 0;
		} else {
			y_min = ((int)pixmap->pos().y() / 1024) * (1 << (32 - _zoom));
		}

		// std::cout << "redrawing tile at " << y_min << std::endl;


		if (is_left) {
			_zones_drawing.draw_zoomed_zone(*tile.bimage, y_min, _zoom, _axis - 1,
			                                &PVZoomedZoneTree::browse_tree_bci_by_y2);
		} else {
			_zones_drawing.draw_zoomed_zone(*tile.bimage, y_min, _zoom, _axis,
			                                &PVZoomedZoneTree::browse_tree_bci_by_y1);
		}
		pixmap->setPixmap(QPixmap::fromImage(tile.bimage->qimage()));
	}

private:
	PVZonesDrawing  &_zones_drawing;
	PVCol            _axis;
	int              _zoom;
	int              _first_tile;

	PVZoomedTile    *_left_tiles;
	PVZoomedTile    *_right_tiles;
	int              _translation_start_y;
};

}

#endif // PVAPRALLELVIEW_PVZOOMEDTILER_H
