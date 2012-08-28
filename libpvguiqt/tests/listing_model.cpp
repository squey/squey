#include <pvkernel/core/picviz_intrin.h>
#include <pvkernel/core/PVDataTreeObject.h>

#include <picviz/PVMapped.h>
#include <picviz/PVPlotted.h>
#include <picviz/PVSource.h>
#include <picviz/PVView.h>

#include <pvguiqt/PVListingModel.h>

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

	// Get a Picviz tree from the given file/format
	Picviz::PVRoot_p root;
	Picviz::PVSource_sp src = get_src_from_file(root, argv[1], argv[2]);
	src->create_default_view();
	init_random_colors(*src->current_view());

	// Qt app
	QApplication app(argc, argv);

	Picviz::PVView_sp view = src->current_view()->shared_from_this();
	PVGuiQt::PVListingModel* model = new PVGuiQt::PVListingModel(view);

	QTableView* qt_view = new QTableView();
	qt_view->setModel(model);

	QMainWindow* mw = new QMainWindow();
	mw->setCentralWidget(qt_view);

	mw->show();

	// Remove listing when pressing enter
	boost::thread key_thread([&]
		{
			std::cerr << "Press enter to remove data-tree..." << std::endl;
			while (getchar() != '\n');
			std::cout << "view: " << view.get() << std::endl;
			view->remove_from_tree();
			view.reset();
		}
	);

	int ret = app.exec();
	key_thread.join();


	return ret;
}
