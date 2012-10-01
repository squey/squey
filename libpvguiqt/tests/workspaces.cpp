/**
 * \file workspaces.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */


#include <pvparallelview/common.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVBCIBackendImage.h>
#include <pvparallelview/PVBCIDrawingBackendCUDA.h>
#include <pvparallelview/PVZonesDrawing.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVLinesView.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVLibView.h>

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
#include <pvguiqt/PVWorkspace.h>
#include <pvguiqt/PVWorkspacesTabWidget.h>

#include <pvguiqt/PVListingModel.h>
#include <pvguiqt/PVListingSortFilterProxyModel.h>
#include <pvguiqt/PVListingView.h>



#include "common.h"
#include "test-env.h"

#include <iostream>

#include <QApplication>
#include <QMainWindow>
#include <QTabWidget>
#include <QStyle>
#include <QDesktopWidget>
#include <QLabel>
#include <QPushButton>

class CustomMainWindow : public QMainWindow
{
public:

	CustomMainWindow()
	{
		setMinimumSize(1800, 1150);

		setGeometry(
		    QStyle::alignedRect(
		        Qt::LeftToRight,
		        Qt::AlignCenter,
		        size(),
		        qApp->desktop()->availableGeometry()
		    ));
	}
};


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
	Picviz::PVSource_sp src2 = get_src_from_file(root->get_children().at(0), argv[1], argv[2]);
	src->create_default_view();
	src2->create_default_view();

	Picviz::PVView_p view(src->current_view()->get_parent()->shared_from_this());
	view->process_parent_plotted();

	// Qt app
	QApplication app(argc, argv);

	// Create our model and view
	root->dump();
	src->dump();

	/*PVGuiQt::PVRootTreeModel* model = new PVGuiQt::PVRootTreeModel(*root);
	PVGuiQt::PVRootTreeView* data_tree_display = new PVGuiQt::PVRootTreeView(model);*/

	CustomMainWindow* mw = new CustomMainWindow();

	PVGuiQt::PVWorkspacesTabWidget* workspaces_tab_widget = new PVGuiQt::PVWorkspacesTabWidget(mw);
	workspaces_tab_widget->resize(mw->size());

	PVGuiQt::PVWorkspace* workspace1 = new PVGuiQt::PVWorkspace(src);


	workspaces_tab_widget->addTab(workspace1, "Workspace1");


	PVParallelView::common::init<PVParallelView::PVBCIDrawingBackendCUDA>();

	PVParallelView::PVLibView* plib_view = PVParallelView::common::get_lib_view(*view);

	//QWidget* parallel_view = plib_view->create_view();

	//workspace1->setCentralWidget(parallel_view);


	PVGuiQt::PVListingModel* listing_model = new PVGuiQt::PVListingModel(view);
	PVGuiQt::PVListingSortFilterProxyModel* proxy_model = new PVGuiQt::PVListingSortFilterProxyModel(view);
	proxy_model->setSourceModel(listing_model);
	PVGuiQt::PVListingView* listing_view = new PVGuiQt::PVListingView(view);
	listing_view->setModel(proxy_model);


	//workspace1->setCentralWidget(listing_view);
	workspace1->set_central_display(listing_view, "Listing");

	mw->show();

	return app.exec();
}
