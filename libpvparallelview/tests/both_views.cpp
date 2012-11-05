/**
 * \file both_views.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <QtGui>
#include <QGLWidget>
#include <iostream>

#include <pvparallelview/common.h>
#include <pvkernel/core/picviz_bench.h>
#include <picviz/PVPlotted.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVBCIBackendImage.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVZonesDrawing.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVLinesView.h>
#include <pvparallelview/PVZoomedParallelScene.h>

#include <pvparallelview/PVFullParallelScene.h>
#include <pvparallelview/PVFullParallelView.h>

//#include <pvguiqt/PVAxesCombinationDialog.h>

#include <pvparallelview/PVLibView.h>

#include <pvbase/general.h>

#include "common.h"
#include "zoom_dlg.h"

#include <QApplication>

#include <mcheck.h>

int main(int argc, char** argv)
{
	if (argc < 2) {
		usage(argv[0]);
		return 1;
	}

	QApplication app(argc, argv);

	PVParallelView::common::init_cuda();
	PVLOG_INFO("Pipeline in %p\n", &PVParallelView::common::pipeline());
	PVParallelView::PVLibView* plib_view = create_lib_view_from_args(argc, argv);
	if (plib_view == NULL) {
		return 1;
	}
	PVParallelView::PVFullParallelView* fpv = plib_view->create_view();
	fpv->show();

	ZoomDlg* zdlg = new ZoomDlg(*plib_view);
	zdlg->show();

	{
		/*Picviz::PVView_sp view_sp = plib_view->lib_view()->shared_from_this();
		PVGuiQt::PVAxesCombinationDialog* axes_dlg = new PVGuiQt::PVAxesCombinationDialog(view_sp);
		axes_dlg->show();*/
	}

	app.exec();

	PVParallelView::common::release();


	return 0;
}
