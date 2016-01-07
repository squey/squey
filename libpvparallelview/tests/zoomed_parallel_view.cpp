/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <iostream>

#include <pvkernel/core/inendi_bench.h>

#include <inendi/PVAxis.h>
#include <inendi/PVPlotted.h>
#include <inendi/PVView.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVBCIBackendImage.h>
#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVZonesDrawing.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVZoomedParallelScene.h>
#include <pvparallelview/PVZoomedParallelView.h>

#include <pvbase/general.h>

#include "common.h"

#include <QApplication>
#include <QGraphicsView>

/*****************************************************************************/

#define CRAND() (127 + (random() & 0x7F))

//#define CONE

void init_rand_plotted(Inendi::PVPlotted::plotted_table_t& p,
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

#define RENDERING_BITS PARALLELVIEW_ZZT_BBITS

int main(int argc, char** argv)
{
	if (argc < 2) {
		usage(argv[0]);
		return 1;
	}

	QApplication app(argc, argv);

	PVParallelView::common::RAII_cuda_init cuda_resources;

	PVParallelView::PVLibView* plib_view = create_lib_view_from_args(argc, argv);
	PVParallelView::PVZoomedParallelView* zpview = plib_view->create_zoomed_view(1);
	zpview->resize(1024, 1024);
	zpview->show();

	app.exec();

	return 0;
}
