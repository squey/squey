/**
 * \file zoomed_parallel_scene.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <Qt>
#include <QObject>
#include <QtGui>
#include <QGLWidget>
#include <iostream>

#include <pvparallelview/common.h>
#include <pvkernel/core/picviz_bench.h>
#include <picviz/PVPlotted.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVBCIBackendImage.h>
#include <pvparallelview/PVBCIDrawingBackendCUDA.h>
#include <pvparallelview/PVZonesDrawing.h>
#include <pvparallelview/PVZonesManager.h>

#include <QApplication>

#include <pvparallelview/PVAxisGraphicsItem.h>
#include <picviz/PVAxis.h>

#include "zoomed_parallel_scene.h"

static int TILE_NUMBER = 3;

#define ZOOM_STEPS 5

#define TILE_OFFSET (PARALLELVIEW_AXIS_WIDTH / 2)

#define TILE_SIZE (IMAGE_HEIGHT / 2)

#define ROOT5_2 1.148698355

#define SCENE_HEIGHT 1024

int get_zoom_level(int z)
{
	return z / ZOOM_STEPS;
}

int get_zoom_step(int z)
{
	return z % ZOOM_STEPS;
}

double compute_zoom(int z)
{
	return pow(2, get_zoom_level(z)) * pow(ROOT5_2, get_zoom_step(z));
}

/* TODO: with a big zoom level, moving fast the scrollbar until it reaches
 *       its limits (sliderToMinimum or sliderToMaximum), the update_view is
 *       not invoked... Qt's fault or mine?
 */

/*****************************************************************************/

struct PVZoomedTile
{
	PVParallelView::PVBCIBackendImage_p  bimage;
	QGraphicsPixmapItem                 *pixmap;
	bool                                 valid;

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

/*****************************************************************************
 * PVZoomedParallelScene::PVZoomedParallelScene
 *****************************************************************************/

PVZoomedParallelScene::PVZoomedParallelScene(QObject *parent,
                                             PVParallelView::PVZonesDrawing &zones_drawing,
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

	connect(view()->verticalScrollBar(), SIGNAL(actionTriggered(int)),
	        this, SLOT(update_view(int)));
	connect(view()->verticalScrollBar(), SIGNAL(sliderReleased()),
	        this, SLOT(update_view()));

	// TODO: use/init position, zoom, _old_level
	//view()->verticalScrollBar()->setValue(position);

	_old_level = get_zoom_level(_wheel_value);

	if(axis > 0) {
		_left_tiles = new PVZoomedTile[TILE_NUMBER];

		for (int i = 0; i < TILE_NUMBER; ++i) {
			PVParallelView::PVBCIBackendImage_p img = zones_drawing.create_image(IMAGE_HEIGHT);
			_left_tiles[i].bimage = img;

			_left_tiles[i].pixmap = addPixmap(QPixmap::fromImage(img->qimage()));
			_left_tiles[i].pixmap->setTransformationMode(Qt::SmoothTransformation);
		}
	}

	if (axis < zones_drawing.get_zones_manager().get_number_cols()) {
		_right_tiles = new PVZoomedTile[TILE_NUMBER];

		for (int i = 0; i < TILE_NUMBER; ++i) {
			PVParallelView::PVBCIBackendImage_p img = zones_drawing.create_image(IMAGE_HEIGHT);
			_right_tiles[i].bimage = img;

			_right_tiles[i].pixmap = addPixmap(QPixmap::fromImage(img->qimage()));
			_right_tiles[i].pixmap->setTransformationMode(Qt::SmoothTransformation);
		}
	}

	QPen pen;
	pen.setColor(QColor(255, 255, 255, 32));
	pen.setWidth(PARALLELVIEW_AXIS_WIDTH);

	_axis_line = addLine(QLineF(0, 0, 0, 1), pen);
	_axis_line->setFlag(QGraphicsItem::ItemIgnoresTransformations, true);

	update_zoom();

	// make sure all tiles are invalid
	invalidate_tiles();
	check_tiles_validity();
}

/*****************************************************************************
 * PVZoomedParallelScene::~PVZoomedParallelScene
 *****************************************************************************/

PVZoomedParallelScene::~PVZoomedParallelScene()
{
	if (_left_tiles) {
		delete [] _left_tiles;
	}
	if (_right_tiles) {
		delete [] _right_tiles;
	}
}

/*****************************************************************************
 * PVZoomedParallelScene::mouseMoveEvent
 *****************************************************************************/

void PVZoomedParallelScene::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
	if (event->buttons() == Qt::RightButton) {
		QScrollBar *sb = view()->verticalScrollBar();
		long offset = (long)(_translation_start_y - event->scenePos().y());
		sb->setValue(sb->value() + offset);
		update_view();
	}
}

/*****************************************************************************
 * PVZoomedParallelScene::mousePressEvent
 *****************************************************************************/

void PVZoomedParallelScene::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
	if (event->button() == Qt::RightButton) {
		_translation_start_y = event->scenePos().y();
	}
}

/*****************************************************************************
 * PVZoomedParallelScene::mouseReleaseEvent
 *****************************************************************************/

void PVZoomedParallelScene::mouseReleaseEvent(QGraphicsSceneMouseEvent */*event*/)
{
	// do nothing to avoid the behaviour of QGraphicsScene::mouseReleaseEvent()
}

/*****************************************************************************
 * PVZoomedParallelScene::wheelEvent
 *****************************************************************************/

void PVZoomedParallelScene::wheelEvent(QGraphicsSceneWheelEvent* event)
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
 * PVZoomedParallelScene::update_tile_position
 *****************************************************************************/

void PVZoomedParallelScene::update_tile_position(int tile_index)
{
	/* some init
	 */
	int level = get_zoom_level(_wheel_value);
	long tile_num = 1 << (1 + level); // real formula is: 2 * (1 << level);
	double tile_scale = 1. / (double)tile_num;
	double tile_height = SCENE_HEIGHT / (double)tile_num;

	/* get the position of the upper tile (in tile space). It is
	 * deduced from the view's top in scene space
	 */
	long tile_upper_pos = (long)(view()->mapToScene(0, 0).y() / tile_height);

	/* if the processed tile has a lower index that the upper one, it
	 * must be shifted downward
	 */
	long tile_shift = tile_index - (tile_upper_pos % TILE_NUMBER);
	if (tile_shift < 0) {
		tile_shift += TILE_NUMBER;
	}

	double old_y;
	double new_y = (tile_upper_pos + tile_shift) * tile_height;

	QTransform t;
	t.scale(0.5, tile_scale);

	double offset = 2. * tile_scale;

	if (_left_tiles) {
		PVZoomedTile &tile = _left_tiles[tile_index];

		tile.set_visibility(level);

		old_y = tile.pixmap->y();

		tile.pixmap->setX(-512 - offset);
		tile.pixmap->setY(new_y);
		tile.pixmap->setTransform(t);

		tile.valid = (old_y == new_y);
	}

	if (_right_tiles) {
		PVZoomedTile &tile = _right_tiles[tile_index];
		tile.set_visibility(level);

		old_y = tile.pixmap->y();

		tile.pixmap->setX(offset);
		tile.pixmap->setY(new_y);
		tile.pixmap->setTransform(t);

		tile.valid = (old_y == new_y);
	}
}

/*****************************************************************************
 * PVZoomedParallelScene::update_zoom
 *****************************************************************************/

void PVZoomedParallelScene::update_zoom()
{
	int level = get_zoom_level(_wheel_value);

	setSceneRect(-SCENE_HEIGHT * 0.5, 0., SCENE_HEIGHT, SCENE_HEIGHT);

	// Phillipe's magical formula
	double s = compute_zoom(_wheel_value);

	std::cout << "::update_zoom(...)" << std::endl;
	std::cout << "    _wheel_value: " << _wheel_value << std::endl;
	std::cout << "    s           : " << s << std::endl;
	view()->resetTransform();
	view()->scale(s, s);

	// qreal ncy = view()->mapToScene(view()->viewport()->rect()).boundingRect().center().y();
	// view()->centerOn(0., ncy);

	update_axis_line();
	update_view();

	if (level != _old_level) {
		_old_level = level;
		invalidate_tiles();
	}
}

/*****************************************************************************
 * PVZoomedParallelScene::invalidate_tiles
 *****************************************************************************/

void PVZoomedParallelScene::invalidate_tiles()
{
	if (_left_tiles) {
		for (int i = 0; i < TILE_NUMBER; ++i) {
			_left_tiles[i].valid = false;
		}
	}

	if (_right_tiles) {
		for (int i = 0; i < TILE_NUMBER; ++i) {
			_right_tiles[i].valid = false;
		}
	}
}

/*****************************************************************************
 * PVZoomedParallelScene::check_tiles_validity
 *****************************************************************************/

void PVZoomedParallelScene::check_tiles_validity()
{
	if (_left_tiles) {
		for (int i = 0; i < TILE_NUMBER; ++i) {
			PVZoomedTile &tile = _left_tiles[i];
			if (tile.valid == false) {
				render_tile(tile, true);
			}
		}
	}

	if (_right_tiles) {
		for (int i = 0; i < TILE_NUMBER; ++i) {
			PVZoomedTile &tile = _right_tiles[i];
			if (tile.valid == false) {
				render_tile(tile, false);
			}
		}
	}
}

/*****************************************************************************
 * PVZoomedParallelScene::render_tile
 *****************************************************************************/

void PVZoomedParallelScene::render_tile(PVZoomedTile &tile, bool is_left)
{
	QGraphicsPixmapItem *pixmap = tile.pixmap;
	int level = get_zoom_level(_wheel_value);

	if (tile.set_visibility(level) == false) {
		return;
	}

	long tile_num = 1 << (1 + level);
	double tile_height = SCENE_HEIGHT / (double)tile_num;

	// we render at level+1
	level += 1;

	uint32_t y_min = ((long)pixmap->y() / tile_height) * (1 << (32 - level));

	BENCH_START(render);
	if (is_left) {
		_zones_drawing.draw_zoomed_zone(*tile.bimage, y_min, level, _axis - 1,
		                                &PVParallelView::PVZoomedZoneTree::browse_tree_bci_by_y2);
	} else {
		_zones_drawing.draw_zoomed_zone(*tile.bimage, y_min, level, _axis,
		                                &PVParallelView::PVZoomedZoneTree::browse_tree_bci_by_y1);
	}
	BENCH_END(render, "render tile", 1, 1, 1, 1);
	pixmap->setPixmap(QPixmap::fromImage(tile.bimage->qimage()));
}


/*****************************************************************************
 * PVZoomedParallelScene::update_axis_line
 *****************************************************************************/

void PVZoomedParallelScene::update_axis_line()
{
	qreal v = view()->verticalScrollBar()->value();

	_axis_line->setLine(0., v - 100000., 0., v + 100000.);
}

/*****************************************************************************
 * PVZoomedParallelScene::update_view
 *****************************************************************************/

void PVZoomedParallelScene::update_view(int /*value*/)
{
	for (int i = 0; i < TILE_NUMBER; ++i) {
		update_tile_position(i);
	}
	check_tiles_validity();

	update_axis_line();
}

/*****************************************************************************/

#define CRAND() (127 + (random() & 0x7F))

void usage(const char* path)
{
	std::cerr << "Usage: " << path << " [plotted_file] [nrows] [ncols]" << std::endl;
}

void init_rand_plotted(Picviz::PVPlotted::plotted_table_t& p, PVRow nrows, PVCol ncols)
{
	srand(0);
	p.clear();
	p.reserve(nrows*ncols);
	for (PVRow i = 0; i < nrows*ncols; i++) {
		p.push_back((float)((double)(rand())/(double)RAND_MAX));
	}
}

/*****************************************************************************/

int main(int argc, char** argv)
{
	if (argc < 2) {
		usage(argv[0]);
		return 1;
	}

	QApplication app(argc, argv);

	PVCol ncols, nrows;
	Picviz::PVPlotted::plotted_table_t plotted;
	QString fplotted(argv[1]);
	if (fplotted == "0") {
		if (argc < 4) {
			usage(argv[0]);
			return 1;
		}
		srand(0);
		nrows = atol(argv[2]);
		ncols = atol(argv[3]);

		init_rand_plotted(plotted, nrows, ncols);
	}
	else
	{
		if (!Picviz::PVPlotted::load_buffer_from_file(plotted, ncols, true, QString(argv[1]))) {
			std::cerr << "Unable to load plotted !" << std::endl;
			return 1;
		}
		nrows = plotted.size()/ncols;
	}

	PVParallelView::PVHSVColor* colors = PVParallelView::PVHSVColor::init_colors(nrows);

	Picviz::PVPlotted::uint_plotted_table_t norm_plotted;
	Picviz::PVPlotted::norm_int_plotted(plotted, norm_plotted, ncols);

	// Zone Manager
	PVParallelView::PVZonesManager &zm = *(new PVParallelView::PVZonesManager());
	zm.set_uint_plotted(norm_plotted, nrows, ncols);
	zm.update_all();

	PVParallelView::PVBCIDrawingBackendCUDA backend_cuda;
	PVParallelView::PVZonesDrawing &zones_drawing = *(new PVParallelView::PVZonesDrawing(zm, backend_cuda, *colors));

	QGraphicsView view;
	view.setViewport(new QWidget());
	view.setScene(new PVZoomedParallelScene(&view, zones_drawing,
	                                        /*zone*/ 1, /*pos*/ 0, /*zoom*/ 0));
	view.resize(1024, 1024);
	view.show();

	app.exec();


	return 0;
}
