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

#include <pvhive/PVObserverCallback.h>
#include <pvhive/PVHive.h>

#include <pvguiqt/PVLayerStackWidget.h>

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
	Inendi::PVRoot_p root;
	Inendi::PVSource& src = get_src_from_file(*root, argv[1], argv[2]);
	src.create_default_view();

	// Qt app
	QApplication app(argc, argv);

	Inendi::PVView_sp view = src.current_view()->shared_from_this();
	view->add_new_layer();

	PVGuiQt::PVLayerStackWidget* ls = new PVGuiQt::PVLayerStackWidget(view);
	PVGuiQt::PVLayerStackWidget* ls2 = new PVGuiQt::PVLayerStackWidget(view);

	QMainWindow* mw = new QMainWindow();
	mw->setCentralWidget(ls);

	QMainWindow* mw2 = new QMainWindow();
	mw2->setCentralWidget(ls2);

	mw->show();
	mw2->show();

	// Register callback event
	auto observer = PVHive::create_observer_callback<Inendi::PVLayerStack>(
	    [](Inendi::PVLayerStack const*) { std::cout << "about to be refreshed." << std::endl; },
	    [](Inendi::PVLayerStack const*) { std::cout << "refreshed." << std::endl; },
	    [](Inendi::PVLayerStack const*) { std::cout << "about to be deleted." << std::endl; });

	PVHive::get().register_observer(
	    view, [=](Inendi::PVView& view) { return &view.get_layer_stack(); }, observer);

	int ret = app.exec();

	return ret;
}
