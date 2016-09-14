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

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVBCIBackendImage.h>
#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVZoomedParallelScene.h>
#include <pvparallelview/PVZoomedParallelView.h>

#include <pvbase/general.h>

#include "common.h"

#include <QApplication>
#include <QGraphicsView>

const std::string filename = TEST_FOLDER "/picviz/heat_line.csv";
const std::string fileformat = TEST_FOLDER "/picviz/heat_line.csv.format";

/*****************************************************************************/

int main(int argc, char** argv)
{
	if (argc < 2) {
		return 1;
	}

	QApplication app(argc, argv);

	PVParallelView::common::RAII_backend_init resources;

	TestEnv env(filename, fileformat);

	PVParallelView::PVLibView* plib_view = env.get_lib_view();
	PVParallelView::PVZoomedParallelView* zpview = plib_view->create_zoomed_view(1);
	zpview->resize(1024, 1024);
	zpview->show();

	app.exec();

	return 0;
}
