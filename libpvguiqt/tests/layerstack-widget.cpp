#include <pvkernel/core/picviz_intrin.h>
#include <pvkernel/core/PVDataTreeObject.h>

#include <picviz/PVMapped.h>
#include <picviz/PVPlotted.h>
#include <picviz/PVSource.h>
#include <picviz/PVView.h>

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

	// Get a Picviz tree from the given file/format
	Picviz::PVRoot_sp root = Picviz::PVRoot::get_root_sp();
	Picviz::PVSource_sp src = get_src_from_file(root, argv[1], argv[2]);
	src->create_default_view();

	// Qt app
	QApplication app(argc, argv);

	Picviz::PVView_sp view = src->current_view()->shared_from_this();
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
	auto observer = PVHive::create_observer_callback<Picviz::PVLayerStack>(
	    [](Picviz::PVLayerStack const*) { std::cout << "about to be refreshed." << std::endl; },
	    [](Picviz::PVLayerStack const*) { std::cout << "refreshed." << std::endl; },
	    [](Picviz::PVLayerStack const*) { std::cout << "about to be deleted." << std::endl; });

	PVHive::get().register_observer(view, [=](Picviz::PVView& view) { return &view.get_layer_stack(); }, observer);
	 
	int ret = app.exec();

	return ret;
}
