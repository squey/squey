/**
 * \file stats_listing.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <pvkernel/core/picviz_intrin.h>
#include <pvkernel/core/PVDataTreeObject.h>

#include <picviz/PVMapped.h>
#include <picviz/PVPlotted.h>
#include <picviz/PVSource.h>
#include <picviz/PVView.h>

#include <pvguiqt/PVListingModel.h>
#include <pvguiqt/PVListingSortFilterProxyModel.h>
#include <pvguiqt/PVListingView.h>
#include <pvguiqt/PVStatsListingWidget.h>

#include <QObject>
#include <QApplication>
#include <QMainWindow>
#include <QTableView>
#include <QVBoxLayout>
#include <QDesktopWidget>
#include <QShortcut>

#include <boost/thread.hpp>

#include "common.h"
#include "test-env.h"

#include "stats_listing.h"

struct CustomMainWindow : public QMainWindow
{
	CustomMainWindow(QWidget* parent = 0) : QMainWindow(parent)
	{
		setGeometry(
			QStyle::alignedRect(
					Qt::LeftToRight,
					Qt::AlignCenter,
					QSize(500, 800),
					QApplication::desktop()->availableGeometry()
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
	src->create_default_view();
	init_random_colors(*src->current_view());

	// Qt app
	QApplication app(argc, argv);

	Picviz::PVView_sp view = src->current_view()->shared_from_this();
	PVGuiQt::PVListingModel* model = new PVGuiQt::PVListingModel(view);
	PVGuiQt::PVListingSortFilterProxyModel* proxy_model = new PVGuiQt::PVListingSortFilterProxyModel(view);
	proxy_model->setSourceModel(model);

	PVGuiQt::PVListingView* qt_view = new PVGuiQt::PVListingView(view);
	qt_view->setModel(proxy_model);


	ViewSlots view_slots(*view);

	view.reset();

	CustomMainWindow* mw = new CustomMainWindow();

	PVGuiQt::PVStatsListingWidget* stats_listing = new PVGuiQt::PVStatsListingWidget(qt_view);
	mw->setCentralWidget(stats_listing);

	mw->show();

	QShortcut* select_all = new QShortcut(QKeySequence(Qt::Key_A), mw);
	select_all->setContext(Qt::ApplicationShortcut);
	QObject::connect(select_all, SIGNAL(activated()), &view_slots, SLOT(select_all()));

	QShortcut* change_axes_comb = new QShortcut(QKeySequence(Qt::Key_B), mw);
	select_all->setContext(Qt::ApplicationShortcut);
	QObject::connect(change_axes_comb, SIGNAL(activated()), &view_slots, SLOT(change_axes_combination()));

	// Remove listing when pressing enter
	boost::thread key_thread([&]
		{
			std::cerr << "Press enter to remove data-tree..." << std::endl;
			while (getchar() != '\n');
			//pview->remove_from_tree();
			root.reset();
		}
	);

	int ret = app.exec();
	key_thread.join();

	return ret;
}
