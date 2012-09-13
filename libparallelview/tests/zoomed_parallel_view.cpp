/**
 * \file zoomed_parallel_scene.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <iostream>

#include <pvkernel/core/picviz_bench.h>

#include <picviz/PVAxis.h>
#include <picviz/PVPlotted.h>
#include <picviz/PVView.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVBCIBackendImage.h>
#include <pvparallelview/PVBCIDrawingBackendCUDA.h>
#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVZonesDrawing.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVZoomedParallelScene.h>
#include <pvparallelview/PVZoomedParallelView.h>

#include <pvbase/general.h>

#include <QApplication>
#include <QGraphicsView>

/*****************************************************************************/

#define CRAND() (127 + (random() & 0x7F))

//#define CONE

void init_rand_plotted(Picviz::PVPlotted::plotted_table_t& p,
                       PVRow nrows, PVCol ncols)
{
	srand(0);
	p.clear();
	p.reserve(nrows*ncols);
#ifdef CONE
	for (PVRow i = 0; i < nrows; i++) {
		p.push_back(0.8f + 0.2f * (i / (float)nrows));
	}
	for (PVRow i = 0; i < nrows; i++) {
		p.push_back(0.9);
	}
#else
	for (PVRow i = 0; i < nrows*ncols; i++) {
		p.push_back((float)((double)(rand())/(double)RAND_MAX));
	}
#endif
}

/*****************************************************************************/

void usage(const char* path)
{
	std::cerr << "Usage: " << path << " [plotted_file] [nrows] [ncols]"
	          << std::endl;
}

/*****************************************************************************/

#define RENDERING_BITS PARALLELVIEW_ZZT_BBITS

typedef PVParallelView::PVZonesDrawing<RENDERING_BITS> zones_drawing_t;


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

		if (nrows > PICVIZ_LINES_MAX) {
			std::cerr << "nrows is too big (max is " << PICVIZ_LINES_MAX << ")" << std::endl;
			return 1;
		}

		ncols = atol(argv[3]);

#ifdef CONE
	nrows = 16;
	ncols = 2;
#endif

		init_rand_plotted(plotted, nrows, ncols);
	} else {
		if (!Picviz::PVPlotted::load_buffer_from_file(plotted, ncols,
		                                              true, QString(argv[1]))) {
			std::cerr << "Unable to load plotted !" << std::endl;
			return 1;
		}
		nrows = plotted.size()/ncols;

		if (nrows > PICVIZ_LINES_MAX) {
			std::cerr << "nrows is too big (max is " << PICVIZ_LINES_MAX << ")" << std::endl;
			return 1;
		}
	}

	Picviz::PVPlotted::uint_plotted_table_t norm_plotted;
	Picviz::PVPlotted::norm_int_plotted(plotted, norm_plotted, ncols);

	Picviz::PVView_sp fake_view(new Picviz::PVView());
	fake_view->reset_layers();

	PVParallelView::common::init<PVParallelView::PVBCIDrawingBackendCUDA>();
	PVParallelView::PVLibView* plib_view = PVParallelView::common::get_lib_view(*fake_view, norm_plotted, nrows, ncols);
	plib_view->get_zones_manager().update_all();

	PVParallelView::PVSlidersManager_p sm_p =
		PVParallelView::PVSlidersManager_p(new PVParallelView::PVSlidersManager);


	PVParallelView::PVZoomedParallelView* zpview = plib_view->create_zoomed_view(1);
	zpview->resize(1024, 1024);
	zpview->show();

	app.exec();

	PVParallelView::common::release();

	return 0;
}
