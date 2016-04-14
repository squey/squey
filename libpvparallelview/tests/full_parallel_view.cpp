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
#include <pvkernel/core/PVSharedPointer.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVBCIBackendImage.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVLinesView.h>

#include <pvparallelview/PVFullParallelScene.h>
#include <pvparallelview/PVFullParallelView.h>

//#include <pvguiqt/PVAxesCombinationDialog.h>
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

	PVParallelView::common::RAII_cuda_init cuda_resources;

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

	{
		Inendi::PVView_sp view_sp = plib_view->lib_view()->shared_from_this();

		dlg = new QDialog();
		dlg->setAttribute(Qt::WA_DeleteOnClose, true);

		layout = new QVBoxLayout(dlg);
		layout->setContentsMargins(0, 0, 0, 0);
		dlg->setLayout(layout);

		/*QWidget *axes = new PVGuiQt::PVAxesCombinationDialog(view_sp);
		layout->addWidget(axes);*/
		dlg->show();
	}

	app.exec();

	// RH: a deadlock in PVLogger
	// PVParallelView::common::release();

	return 0;
}
