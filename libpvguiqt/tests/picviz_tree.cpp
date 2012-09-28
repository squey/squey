#include <pvkernel/core/picviz_intrin.h>
#include <pvkernel/core/PVDataTreeObject.h>
#include <picviz/PVMapped.h>
#include <picviz/PVPlotted.h>
#include <picviz/PVSource.h>
#include <picviz/PVView.h>
#include <pvhive/PVActor.h>
#include <pvhive/PVCallHelper.h>
#include <pvguiqt/PVHiveDataTreeModel.h>
#include <pvguiqt/PVRootTreeModel.h>
#include <pvguiqt/PVRootTreeView.h>

#include <QApplication>
#include <QMainWindow>
#include <QDialog>
#include <QVBoxLayout>

#include <boost/thread.hpp>

#include <iostream>
#include <unistd.h> // for usleep

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
	//Picviz::PVSource_sp src2 = get_src_from_file(root->get_children().at(0), argv[1], argv[2]);
	src->create_default_view();
	//src2->create_default_view();

	//Picviz::PVView_p new_view(src->current_view()->get_parent()->shared_from_this());
	//new_view->process_parent_plotted();

	// Qt app
	QApplication app(argc, argv);

	// Create our model and view
	root->dump();
	src->dump();

	PVGuiQt::PVRootTreeModel* model = new PVGuiQt::PVRootTreeModel(*src);
	PVGuiQt::PVRootTreeView* view = new PVGuiQt::PVRootTreeView(model);
	view->setModel(model);

	//PVGuiQt::PVRootTreeModel* model2 = new PVGuiQt::PVRootTreeModel(*src);
	//PVGuiQt::PVRootTreeView* view2 = new PVGuiQt::PVRootTreeView(model2);

	QMainWindow* mw = new QMainWindow();
	mw->setCentralWidget(view);

	//QMainWindow* mw2 = new QMainWindow();
	//mw2->setCentralWidget(view2);

	mw->show();
	//mw2->show();

	src.reset();
	//src2.reset();
	//new_view.reset();

	// Remove listing when pressing enter
	boost::thread key_thread([&]
		{
			std::cerr << "Press enter to remove data-tree..." << std::endl;
			while (getchar() != '\n');
			root.reset();
		}
	);

	return app.exec();
}
