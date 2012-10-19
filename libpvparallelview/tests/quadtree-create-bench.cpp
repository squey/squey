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
