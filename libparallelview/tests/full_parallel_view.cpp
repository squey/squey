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
#include <pvparallelview/PVLinesView.h>

#include <pvparallelview/PVAxisWidget.h>
#include <picviz/PVAxis.h>

#include <QApplication>

#define CRAND() (127 + (random() & 0x7F))

class PVFullParallelView : public QGraphicsView
{
public:
	virtual void translate_viewport(int translation)
	{
		QScrollBar *hBar = horizontalScrollBar();
		hBar->setValue(hBar->value() + (isRightToLeft() ? -translation : translation));
	}
};

class PVParallelScene : public QGraphicsScene
{
public:
	PVParallelScene(QObject* parent, PVParallelView::PVLinesView* lines_view) : QGraphicsScene(parent), _lines_view(lines_view)
	{
		_lines_view->render_all_imgs(PVParallelView::ImageWidth);
		PVParallelView::PVLinesView::list_zone_images_t images = _lines_view->get_zones_images();

		PVParallelView::PVAxisWidget *axisw;
		Picviz::PVAxis *axis;
		PVZoneID z;
		int pos = 0;

		// Add visible zones
		for (PVZoneID z = 0; z < (PVZoneID) images.size() ; z++) {
			QGraphicsPixmapItem* zone_image = addPixmap(QPixmap::fromImage(images[z].bg->qimage()));
			zone_image->setOpacity(0.5);
			_zones.push_back(zone_image);
			if (z < _lines_view->get_zones_manager().get_number_zones()) {
				zone_image->setPos(QPointF(_lines_view->get_zone_absolute_pos(z), 0));
			}
		}

		// Add ALL axes
		PVZoneID nzones = (PVZoneID) _lines_view->get_zones_manager().get_number_cols();
		for (z = 0; z < nzones; z++) {
			axis = new Picviz::PVAxis();
			axis->set_name(QString("axis ") + QString::number(z));
			axis->set_color(PVCore::PVColor::fromRgba(CRAND(), CRAND(), CRAND(), 0));
			axis->set_titlecolor(PVCore::PVColor::fromRgba(CRAND(), CRAND(), CRAND(), 0));
			_axes.push_back(axis);

			if (z < nzones-1) {
				pos = _lines_view->get_zones_manager().get_zone_absolute_pos(z);
			}
			else {
				// Special case for last axis
				pos += _lines_view->get_zones_manager().get_zone_width(z-1);
			}

			axisw = new PVParallelView::PVAxisWidget(axis);
			axisw->setPos(QPointF(pos - PVParallelView::AxisWidth, 0));
			addItem(axisw);
			axisw->add_range_sliders(768, 1000);
		}
	}

	PVFullParallelView* view()
	{
		return (PVFullParallelView*)parent() ;
	}

	void update_zones_position()
	{
		PVParallelView::PVLinesView::list_zone_images_t images = _lines_view->get_zones_images();
		for (PVZoneID zid = _lines_view->get_first_drawn_zone(); zid <= _lines_view->get_last_drawn_zone(); zid++) {
			const PVZoneID img_id = zid-_lines_view->get_first_drawn_zone();
			_zones[img_id]->setPixmap(QPixmap::fromImage(images[img_id].bg->qimage()));
			_zones[img_id]->setPos(QPointF(_lines_view->get_zone_absolute_pos(zid), 0));
		}

	}

	void mouseMoveEvent(QGraphicsSceneMouseEvent *event)
	{
		if (event->buttons() == Qt::RightButton) {
			// Translate viewport
			QScrollBar *hBar = view()->horizontalScrollBar();
			view()->translate_viewport(_translation_start_x - event->scenePos().x());
		}
	}

	void mousePressEvent(QGraphicsSceneMouseEvent *event)
	{
		if (event->button() == Qt::RightButton)
		{
			// Store view position to compute translation
			_translation_start_x = event->scenePos().x();
		}
	}

	void mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
	{
		if (event->button() == Qt::RightButton)
		{
			// translate zones
			uint32_t view_x = view()->horizontalScrollBar()->value();
			_lines_view->translate(view_x, view()->width());
			update_zones_position();
		}
	}

	void wheelEvent(QGraphicsSceneWheelEvent* event)
	{
		int zoom = event->delta() / 2;

		// Local zoom
		if (event->modifiers() == Qt::ControlModifier) {
			PVZoneID zid = _lines_view->get_zone_from_scene_pos(event->scenePos().x());
			uint32_t z_width = _lines_view->get_zone_width(zid);
			if (_lines_view->set_zone_width_and_render(zid, z_width + zoom)) {
				update_zones_position();
			}
		}
		//Global zoom
		else
		{
			uint32_t view_x = view()->horizontalScrollBar()->value();
			_lines_view->set_all_zones_width_and_render(view_x, view()->width(), [=](uint32_t width){ return width+zoom; });
			update_zones_position();
		}
	}

private:
	PVParallelView::PVLinesView* _lines_view;
    qreal _translation_start_x;

    QList<QGraphicsPixmapItem*> _zones;
    QList<Picviz::PVAxis*> _axes;
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

	PVParallelView::PVLinesView &lines_view = *(new PVParallelView::PVLinesView(zones_drawing, ncols/2));

	PVFullParallelView view;
	view.setViewport(new QWidget());
	view.setScene(new PVParallelScene(&view, &lines_view));
	view.resize(1920, 1600);
	view.horizontalScrollBar()->setValue(0);
	view.show();

	app.exec();


	return 0;
}
