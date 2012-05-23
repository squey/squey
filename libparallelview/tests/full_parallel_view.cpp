#include <QtGui>
#include <QGLWidget>
#include <iostream>

#include <pvkernel/core/picviz_bench.h>
#include <picviz/PVPlotted.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVBCIBackendImage.h>
#include <pvparallelview/PVBCIDrawingBackendCUDA.h>
#include <pvparallelview/PVZonesDrawing.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVLinesView.h>

#include <QApplication>

#define WIDTH 1920
#define HEIGHT 1600

class GraphicsView : public QGraphicsView
{
public:

	virtual void translateViewPort(int translation)
	{
		QScrollBar *hBar = horizontalScrollBar();
		hBar->setValue(hBar->value() + (isRightToLeft() ? -translation : translation));
	}
};

class OpenGLScene : public QGraphicsScene
{
public:
	OpenGLScene(QObject* parent, PVParallelView::PVLinesView* lines_view) : QGraphicsScene(parent), _lines_view(lines_view)
	{
		_lines_view->render_all();

		PVParallelView::PVLinesView::list_zone_images_t images = _lines_view->get_zones_images();

		int pos = 0;
		for (int z = 0; z < images.size() ; z++) {
			QGraphicsPixmapItem* zone_image = addPixmap(QPixmap::fromImage(images[z].all->qimage()));
			zone_image->setPos(QPointF(pos, 0));
			pos += _lines_view->get_zone_width(z) + 3;
		}
	}

	void mouseMoveEvent(QGraphicsSceneMouseEvent *event)
	{
		if (event->buttons() == Qt::RightButton) {
			GraphicsView* view = ((GraphicsView*)parent());

			QScrollBar *hBar = view->horizontalScrollBar();
			view->translateViewPort(_translation_start_x - event->scenePos().x());
		}
	}

	void mousePressEvent(QGraphicsSceneMouseEvent *event)
	{
		if (event->button() == Qt::RightButton)
		{
			GraphicsView* view = ((GraphicsView*)parent());

			_hscroll_value = view->horizontalScrollBar()->value();
			_translation_start_x = event->scenePos().x();
		}
	}

	void mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
	{
		if (event->button() == Qt::RightButton)
		{
			GraphicsView* view = ((GraphicsView*)parent());
			int translation = _hscroll_value - view->horizontalScrollBar()->value();
		}
	}

	/*void wheelEvent(QGraphicsSceneWheelEvent* event)
	{
		int zoom = event->delta();
		if (event->modifiers() == Qt::ControlModifier) {
			_zoom_manager->set_local_zoom(event->scenePos(), zoom);
		}
		else {
			_zoom_manager->set_global_zoom(zoom);
		}

	}*/

	/*void drawBackground(QPainter *painter, const QRectF &)
	{

	}*/

private:
	PVParallelView::PVLinesView* _lines_view;
    unsigned int _hscroll_value;
    qreal _translation_start_x;
};

void usage(const char* path)
{
	std::cerr << "Usage: " << path << " [plotted_file] [nrows] [ncols]" << std::endl;
}

void init_rand_plotted(Picviz::PVPlotted::plotted_table_t& p, PVRow nrows, PVCol ncols)
{
	srand(time(NULL));
	p.clear();
	p.reserve(nrows*ncols);
	for (PVRow i = 0; i < nrows*ncols; i++) {
		p.push_back((float)((double)(rand())/(double)RAND_MAX));
	}
}

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
		srand(time(NULL));
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

	PVParallelView::PVLinesView &lines_view = *(new PVParallelView::PVLinesView(zones_drawing, 5));

	GraphicsView view;
	view.setViewport(new QGLWidget());
	view.setScene(new OpenGLScene(&view, &lines_view));
	view.resize(1920, 1600);
	view.scale(3, 1);
	view.show();

	app.exec();


	return 0;
}
