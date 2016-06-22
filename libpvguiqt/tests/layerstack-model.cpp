/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/inendi_intrin.h>

#include <inendi/PVMapped.h>
#include <inendi/PVPlotted.h>
#include <inendi/PVSource.h>
#include <inendi/PVView.h>
#include <inendi/PVRoot.h>

#include <pvguiqt/PVLayerStackDelegate.h>
#include <pvguiqt/PVLayerStackModel.h>
#include <pvguiqt/PVLayerStackView.h>

#include <QApplication>
#include <QMainWindow>
#include <QTableView>
#include <QVBoxLayout>

#include "common.h"
#include "test-env.h"

int main(int argc, char** argv)
{
	if (argc <= 2) {
		std::cerr << "Usage: " << argv[0] << " file format" << std::endl;
		return 1;
	}
	PVCore::PVIntrinsics::init_cpuid();
	init_env();

	// Get a INENDI tree from the given file/format
	Inendi::PVRoot root;
	Inendi::PVSource& src = get_src_from_file(root, argv[1], argv[2]);
	src.emplace_add_child()   // Mapped
	    .emplace_add_child()  // Plotted
	    .emplace_add_child(); // View

	// Qt app
	QApplication app(argc, argv);

	Inendi::PVView& view = *src.current_view();
	PVGuiQt::PVLayerStackDelegate* delegate = new PVGuiQt::PVLayerStackDelegate(view);
	PVGuiQt::PVLayerStackModel* model = new PVGuiQt::PVLayerStackModel(view);
	PVGuiQt::PVLayerStackModel* model2 = new PVGuiQt::PVLayerStackModel(view);

	PVGuiQt::PVLayerStackView* qt_view = new PVGuiQt::PVLayerStackView();
	PVGuiQt::PVLayerStackView* qt_view2 = new PVGuiQt::PVLayerStackView();
	qt_view->setModel(model);
	qt_view->setItemDelegate(delegate);
	qt_view2->setModel(model2);
	qt_view2->setItemDelegate(delegate);

	QMainWindow* mw = new QMainWindow();
	mw->setCentralWidget(qt_view);

	QMainWindow* mw2 = new QMainWindow();
	mw2->setCentralWidget(qt_view2);

	mw->show();
	mw2->show();

	int ret = app.exec();

	return ret;
}
