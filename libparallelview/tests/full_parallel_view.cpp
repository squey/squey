/**
 * \file full_parallel_view.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <QtGui>
#include <QGLWidget>
#include <iostream>

#include <pvkernel/core/picviz_bench.h>

#include <picviz/PVPlotted.h>
#include <picviz/PVView.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVBCIBackendImage.h>
#include <pvparallelview/PVBCIDrawingBackendCUDA.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVZonesDrawing.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVLinesView.h>

#include <pvparallelview/PVFullParallelScene.h>
#include <pvparallelview/PVFullParallelView.h>

#include <pvkernel/core/PVSharedPointer.h>

#include <pvparallelview/PVLibView.h>

#include <pvbase/general.h>

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

#define RENDERING_BITS 10

int main(int argc, char** argv)
{
	if (argc < 2) {
		usage(argv[0]);
		return 1;
	}

	QApplication app(argc, argv);

	PVCol ncols;
	PVRow nrows;

	Picviz::PVPlotted::uint_plotted_table_t norm_plotted;
	QString fplotted(argv[1]);
	if (fplotted == "0") {
		if (argc < 4) {
			usage(argv[0]);
			return 1;
		}
		srand(time(NULL));
		nrows = atol(argv[2]);

		if (nrows > PICVIZ_LINES_MAX) {
			std::cerr << "nrows is too big (max is " << PICVIZ_LINES_MAX << ")" << std::endl;
			return 1;
		}

		ncols = atol(argv[3]);

		Picviz::PVPlotted::plotted_table_t plotted;
		init_rand_plotted(plotted, nrows, ncols);
		Picviz::PVPlotted::norm_int_plotted(plotted, norm_plotted, ncols);
	}
	else
	{
		bool plotted_uint = false;
		if (argc >= 3) {
			plotted_uint = (argv[2][0] == '1');
		}

		if (plotted_uint) {
			if (!Picviz::PVPlotted::load_buffer_from_file(norm_plotted, nrows, ncols, true, QString(argv[1]))) {
				std::cerr << "Unable to load plotted !" << std::endl;
				return 1;
			}
		}

		else {
			Picviz::PVPlotted::plotted_table_t plotted;
			if (!Picviz::PVPlotted::load_buffer_from_file(plotted, ncols, true, QString(argv[1]))) {
				std::cerr << "Unable to load plotted !" << std::endl;
				return 1;
			}
			nrows = plotted.size()/ncols;
			Picviz::PVPlotted::norm_int_plotted(plotted, norm_plotted, ncols);
		}

		if (nrows > PICVIZ_LINES_MAX) {
			std::cerr << "nrows is too big (max is " << PICVIZ_LINES_MAX << ")" << std::endl;
			return 1;
		}
	}

	//PVCore::PVHSVColor* colors = PVCore::PVHSVColor::init_colors(nrows);

	Picviz::PVView_sp fake_view(new Picviz::PVView());
	fake_view->reset_layers();

	PVParallelView::common::init<PVParallelView::PVBCIDrawingBackendCUDA>();
	PVParallelView::PVLibView* plib_view = PVParallelView::common::get_lib_view(*fake_view, norm_plotted, nrows, ncols);
	plib_view->get_zones_manager().update_all();
	plib_view->create_view();

	app.exec();

	PVParallelView::common::release();

	return 0;
}
