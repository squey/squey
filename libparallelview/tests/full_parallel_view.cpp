/**
 * \file full_parallel_view.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <QtGui>
#include <QGLWidget>
#include <iostream>

#include <pvkernel/core/picviz_bench.h>
#include <pvkernel/core/PVSharedPointer.h>

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

#include <pvparallelview/PVLibView.h>

#include <pvbase/general.h>

#include <QApplication>

#include "common.h"

#define RENDERING_BITS 10

int main(int argc, char** argv)
{
	if (argc < 2) {
		usage(argv[0]);
		return 1;
	}

	QApplication app(argc, argv);

	PVParallelView::common::init<PVParallelView::PVBCIDrawingBackendCUDA>();

	QDialog *dlg = new QDialog();
	dlg->setAttribute(Qt::WA_DeleteOnClose, true);

	QLayout *layout = new QVBoxLayout(dlg);
	layout->setContentsMargins(0, 0, 0, 0);
	dlg->setLayout(layout);

	PVParallelView::PVLibView* plib_view = create_lib_view_from_args(argc, argv);
	if (plib_view == NULL) {
		return 1;
	}

	QWidget *view = plib_view->create_view();
	layout->addWidget(view);
	dlg->show();

	app.exec();

	// RH: a deadlock in PVLogger
	// PVParallelView::common::release();

	return 0;
}
