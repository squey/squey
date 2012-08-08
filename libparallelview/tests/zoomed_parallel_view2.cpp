/**
 * \file zoomed_parallel_scene.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <iostream>

#include <pvkernel/core/picviz_bench.h>

#include <picviz/PVAxis.h>
#include <picviz/PVPlotted.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVBCIBackendImage.h>
#include <pvparallelview/PVBCIDrawingBackendCUDA.h>
#include <pvparallelview/PVZonesDrawing.h>
#include <pvparallelview/PVZonesManager.h>

#include <QApplication>
#include <QGraphicsView>

#include <pvkernel/core/PVAlgorithms.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVZonesDrawing.h>

#include <QGraphicsSceneMouseEvent>
#include <QGraphicsSceneWheelEvent>

#include <QScrollBar>

#include <limits.h>

/*****************************************************************************/

#define CRAND() (127 + (random() & 0x7F))

void init_rand_plotted(Picviz::PVPlotted::plotted_table_t& p,
                       PVRow nrows, PVCol ncols)
{
	srand(0);
	p.clear();
	p.reserve(nrows*ncols);
	for (PVRow i = 0; i < nrows*ncols; i++) {
		p.push_back((float)((double)(rand())/(double)RAND_MAX));
	}
}

/*****************************************************************************/

void usage(const char* path)
{
	std::cerr << "Usage: " << path << " [plotted_file] [nrows] [ncols]"
	          << std::endl;
}

/*****************************************************************************/

#define print_rect(R) _print_rect(#R, R)

template <typename T>
void _print_rect(const char* text, T r)
{
	std::cout << text << ": "
	          << r.x() << " " << r.y() << " "
	          << r.width() << " " << r.height() << std::endl;
}

/*****************************************************************************/

int rround(const double &d)
{
	return (int)(d + 0.5);
}

/*****************************************************************************/

#define RENDERING_BITS PARALLELVIEW_ZZT_BBITS

typedef PVParallelView::PVZonesDrawing<RENDERING_BITS> zones_drawing_t;

class PVZoomedParallelScene : public QGraphicsScene
{
	constexpr static size_t bbits = PARALLELVIEW_ZZT_BBITS;
	constexpr static int zoom_steps = 10;
	constexpr static double root_step = pow(2.0, 1.0 / zoom_steps);
	constexpr static int tile_number = 3;
	constexpr static uint32_t image_width = 512;
	constexpr static uint32_t image_height = PVParallelView::constants<bbits>::image_height;
	constexpr static int max_wheel_value = 20 * zoom_steps;


public:
	typedef PVParallelView::PVZonesDrawing<bbits> zones_drawing_t;
	typedef typename zones_drawing_t::backend_image_p_t backend_image_p_t;

public:
	PVZoomedParallelScene(QWidget *parent,
	                      zones_drawing_t &zones_drawing,
	                      PVCol axis) :
		QGraphicsScene(parent),
		_zones_drawing(zones_drawing), _axis(axis)
	{
		setBackgroundBrush(Qt::black);

		view()->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
		view()->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
		view()->setResizeAnchor(QGraphicsView::AnchorViewCenter);
		view()->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
		view()->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);

		_wheel_value = 0;

		setSceneRect(-512, 0, 1024, 1024);

		update_zoom();

		if (axis > 0) {
			_left_image = zones_drawing.create_image(image_width);
		}

		if (axis < zones_drawing.get_zones_manager().get_number_zones()) {
			_right_image = zones_drawing.create_image(image_width);
		}
	}

	// kill default behaviour of QGraphicsScene's buttons
	void mousePressEvent(QGraphicsSceneMouseEvent */*event*/)
	{}

	void mouseReleaseEvent(QGraphicsSceneMouseEvent */*event*/)
	{}

	void mouseMoveEvent(QGraphicsSceneMouseEvent */*event*/)
	{}

	void wheelEvent(QGraphicsSceneWheelEvent* event)
	{
		if (event->modifiers() == Qt::ControlModifier) {
			// zoom
			if (event->delta() > 0) {
				if (_wheel_value < max_wheel_value) {
					++_wheel_value;
					update_zoom();
				}
			} else {
				if (_wheel_value > 0) {
					--_wheel_value;
					update_zoom();
				}
			}
		} else if (event->modifiers() == Qt::ShiftModifier) {
			// precise panning
			QScrollBar *sb = view()->verticalScrollBar();
			if (event->delta() > 0) {
				int v = sb->value();
				if (v > sb->minimum()) {
					sb->setValue(v - 1);
				}
			} else {
				int v = sb->value();
				if (v < sb->maximum()) {
					sb->setValue(v + 1);
				}
			}
		} else if (event->modifiers() == Qt::NoModifier) {
			// default panning
			QScrollBar *sb = view()->verticalScrollBar();
			if (event->delta() > 0) {
				sb->triggerAction(QAbstractSlider::SliderSingleStepSub);
			} else {
				sb->triggerAction(QAbstractSlider::SliderSingleStepAdd);
			}
		}

		event->accept();
	}

	virtual void drawBackground(QPainter *painter, const QRectF &/*rect*/)
	{
		QRect screen_rect = view()->viewport()->rect();
		int screen_center = screen_rect.width() / 2;

		std::cout << "=============================" << std::endl;

		_back_image = QImage(screen_rect.size(), QImage::Format_ARGB32);
		_back_image.fill(Qt::black);

		QRectF screen_rect_s = view()->mapToScene(screen_rect).boundingRect();
		print_rect(screen_rect_s);

		QRectF view_rect = sceneRect().intersected(screen_rect_s);
		print_rect(view_rect);

		// true formula: UINT32_MAX * (x / 1024.0)
		uint32_t y_min = view_rect.top() * (UINT32_MAX >> 10);
		uint32_t y_max = view_rect.bottom() * (UINT32_MAX >> 10);

		std::cout << "_zoom_level: " << _zoom_level << std::endl;
		std::cout << "render from " << y_min << " to " << y_max
		          << " (" << y_max - y_min << ")" << std::endl;

		/* TODO: write the zones rendering
		 */

		// we need a painter to draw in _back_image
		QPainter image_painter(&_back_image);

		int step = get_zoom_step();

		int gap_y = (screen_rect_s.top() < 0)?round(-screen_rect_s.top()):0;
		std::cout << "gap_y: " << gap_y << std::endl;
		double alpha = 0.5 * pow(root_step, step);
		double beta = 1 / get_scale_factor();

		if (_left_image.get() != nullptr) {
			_zones_drawing.draw_zoomed_zone(*_left_image, y_min, _zoom_level, _axis - 1,
			                                &PVParallelView::PVZoomedZoneTree::browse_tree_bci_by_y2,
			                                alpha, beta);
			int gap_x = - PARALLELVIEW_AXIS_WIDTH / 2;

			image_painter.drawImage(QPoint(screen_center - gap_x - image_width, gap_y),
			                        _left_image->qimage());
		}

		if (_right_image.get() != nullptr) {
			_zones_drawing.draw_zoomed_zone(*_right_image, y_min, _zoom_level, _axis,
			                                &PVParallelView::PVZoomedZoneTree::browse_tree_bci_by_y1,
			                                alpha, beta);

			int value = 1 + screen_center + PARALLELVIEW_AXIS_WIDTH / 2;

			image_painter.drawImage(QPoint(value, gap_y),
			                        _right_image->qimage());
		}

		// we had to save the painter's state to restore it later
		// the scene transformation matrix is unneeded
		QTransform t = painter->transform();
		painter->resetTransform();

		// the pen has to be saved too
		QPen old_pen = painter->pen();

		// TODO: do the image stuff
		painter->drawImage(QPoint(0,0), _back_image);

		// draw axis
		QPen new_pen = QPen(Qt::white);
		new_pen.setColor(QColor(0xFFFFFFFF));
		new_pen.setWidth(PARALLELVIEW_AXIS_WIDTH);
		painter->setPen(new_pen);
		painter->drawLine(screen_center, 0, screen_center, screen_rect.height());

		// get back the painter's original state
		painter->setTransform(t);
		painter->setPen(old_pen);
	}

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

	int get_tile_num()
	{
		if (bbits == 10) {
			// the true formula is: 2 * (1 << _zoom_level);;
			return 1 << (1 + _zoom_level);
		} else {
			return 1 << (_zoom_level);
		}
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
		nrows = atol(argv[2]);
		ncols = atol(argv[3]);

		init_rand_plotted(plotted, nrows, ncols);
	} else {
		if (!Picviz::PVPlotted::load_buffer_from_file(plotted, ncols,
		                                              true, QString(argv[1]))) {
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

	PVParallelView::PVBCIDrawingBackendCUDA<RENDERING_BITS> backend_cuda;
	zones_drawing_t &zones_drawing = *(new zones_drawing_t(zm, backend_cuda, *colors));

	QGraphicsView view;
	view.setViewport(new QWidget());
	view.setScene(new PVZoomedParallelScene(&view, zones_drawing,
	                                        /*axis*/ 1));
	view.resize(1024, 1024);
	view.show();

	app.exec();


	return 0;
}
