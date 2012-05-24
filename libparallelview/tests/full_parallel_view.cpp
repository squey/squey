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

#include <pvparallelview/PVParallelScene.h>
#include <pvparallelview/PVFullParallelView.h>

#include <QApplication>

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

	PVParallelView::PVFullParallelView view;
	view.setViewport(new QWidget());
	view.setScene(new PVParallelView::PVParallelScene(&view, &lines_view));
	view.resize(1920, 1600);
	view.horizontalScrollBar()->setValue(0);
	view.show();

	app.exec();


	return 0;
}
