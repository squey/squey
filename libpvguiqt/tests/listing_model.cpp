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

#include <pvguiqt/PVListingModel.h>
#include <pvguiqt/PVListingView.h>

#include <QApplication>
#include <QMainWindow>
#include <QTableView>
#include <QVBoxLayout>

#include <boost/thread.hpp>

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

	Inendi::PVView_sp view = src.current_view()->shared_from_this();
	PVGuiQt::PVListingModel* model = new PVGuiQt::PVListingModel(view);

	PVGuiQt::PVListingView* qt_view = new PVGuiQt::PVListingView(view);
	qt_view->setModel(model);

	view.reset();

	QMainWindow* mw = new QMainWindow();
	mw->setCentralWidget(qt_view);

	mw->show();

	// Remove listing when pressing enter
	boost::thread key_thread([&] {
		std::cerr << "Press enter to remove data-tree..." << std::endl;
		while (getchar() != '\n')
			;
	});

	int ret = app.exec();
	key_thread.join();

	return ret;
}
