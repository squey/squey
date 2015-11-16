/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <QtWidgets>
#include <QGLWidget>
#include <iostream>

#include <pvkernel/core/inendi_bench.h>

#include <inendi/PVPlotted.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVZoomedParallelScene.h>
#include <pvparallelview/PVParallelView.h>

#include <pvparallelview/PVLibView.h>

#include <pvbase/general.h>

#include "common.h"

#include <QApplication>

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

	plib_view->request_zoomed_zone_trees(0);

	PVParallelView::common::release();

	return 0;
}
