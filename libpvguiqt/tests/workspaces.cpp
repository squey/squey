/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/inendi_intrin.h>

#include <inendi/PVRoot.h>
#include <inendi/PVMapped.h>
#include <inendi/PVPlotted.h>
#include <inendi/PVSource.h>
#include <inendi/PVView.h>

#include <pvhive/PVActor.h>
#include <pvhive/PVCallHelper.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVParallelView.h>

#include <pvguiqt/common.h>
#include <pvguiqt/PVOpenWorkspacesWidget.h>
#include <pvguiqt/PVWorkspace.h>
#include <pvguiqt/PVWorkspacesTabWidget.h>

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
	Inendi::PVSource_sp src = get_src_from_file(root, argv[1], argv[2]);
	src->create_default_view();

	Inendi::PVView_p view(new Inendi::PVView());
	view->set_parent(src->current_view()->get_parent()->shared_from_this());
	view->process_parent_plotted();

	// Qt app
	QApplication app(argc, argv);

	PVParallelView::common::RAII_cuda_init cuda_resources;
	PVGuiQt::common::register_displays();

	// Create our model and view
	root->dump();
	src->dump();

	PVGuiQt::PVSceneWorkspacesTabWidget* workspaces_tab_widget =
	    new PVGuiQt::PVSceneWorkspacesTabWidget(*src->get_parent());

	PVGuiQt::PVSourceWorkspace* workspace = new PVGuiQt::PVSourceWorkspace(src.get());
	workspaces_tab_widget->addTab(workspace, "Workspace1");
	workspaces_tab_widget->show();

	PVGuiQt::PVOpenWorkspacesWidget* open_workspace =
	    new PVGuiQt::PVOpenWorkspacesWidget(root.get(), NULL);
	open_workspace->show();

	return app.exec();
}
