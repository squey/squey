#include <pvkernel/core/picviz_intrin.h>
#include <pvkernel/core/PVDataTreeObject.h>

#include <picviz/PVMapped.h>
#include <picviz/PVPlotted.h>
#include <picviz/PVSource.h>
#include <picviz/PVView.h>

#include <pvhive/PVObserverCallback.h>

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

	// Get a Picviz tree from the given file/format
	Picviz::PVRoot_p root;
	Picviz::PVSource_sp src = get_src_from_file(root, argv[1], argv[2]);
	src->create_default_view();

	// Qt app
	QApplication app(argc, argv);

	Picviz::PVView_sp view = src->current_view()->shared_from_this();
	PVGuiQt::PVLayerStackDelegate* delegate = new PVGuiQt::PVLayerStackDelegate(*view);
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

	// Register callback event
	auto observer = PVHive::create_observer_callback<Picviz::PVLayerStack>(
	    [](Picviz::PVLayerStack const*) { std::cout << "about to be refreshed." << std::endl; },
	    [](Picviz::PVLayerStack const*) { std::cout << "refreshed." << std::endl; },
	    [](Picviz::PVLayerStack const*) { std::cout << "about to be deleted." << std::endl; });

	PVHive::get().register_observer(view, [=](Picviz::PVView& view) { return &view.get_layer_stack(); }, observer);
	    
	int ret = app.exec();

	return ret;
}
