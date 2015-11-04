/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <QtWidgets>
#include <QGLWidget>
#include <iostream>

#include <pvparallelview/common.h>
#include <pvkernel/core/picviz_bench.h>
#include <picviz/PVPlotted.h>
#include <picviz/PVSelection.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVBCIBackendImage.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVZonesDrawing.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVLinesView.h>
#include <pvparallelview/PVLibView.h>

#include <pvparallelview/PVFullParallelScene.h>
#include <pvparallelview/PVFullParallelView.h>

#include <pvkernel/core/PVSharedPointer.h>

#include <pvbase/general.h>

#include <QApplication>

#include "common.h"

int main(int argc, char** argv)
{
	if (argc < 2) {
		usage(argv[0]);
		return 1;
	}

	QApplication app(argc, argv);

	PVParallelView::common::init_cuda();

	PVParallelView::PVLibView* plib_view = create_lib_view_from_args(argc, argv);
	if (plib_view == NULL) {
		return 1;
	}
	plib_view->create_view();
	plib_view->create_view();

	app.exec();

	PVParallelView::common::release();

	return 0;
}
