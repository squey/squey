/**
 * \file PVWorkspace.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef __PVGUIQT_PVWORKSPACE_H__
#define __PVGUIQT_PVWORKSPACE_H__

#include <iostream>

#include <QMainWindow>
#include <QWidget>
#include <QList>
#include <QDockWidget>
#include <QEvent>
#include <QPushButton>
#include <QHBoxLayout>
#include <QToolBar>
#include <pvkernel/core/PVProgressBox.h>

#include <picviz/PVSource.h>

#include <pvparallelview/PVFullParallelView.h>

#include <pvguiqt/PVViewDisplay.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVLibView.h>

#include <pvguiqt/PVRootTreeModel.h>
#include <pvguiqt/PVRootTreeView.h>

#include <pvguiqt/PVListingModel.h>
#include <pvguiqt/PVListingSortFilterProxyModel.h>
#include <pvguiqt/PVListingView.h>

Q_DECLARE_METATYPE(Picviz::PVView*)

namespace PVGuiQt
{

class PVWorkspace : public QMainWindow
{
	Q_OBJECT;

	friend class PVViewDisplay;
public:
	PVWorkspace(Picviz::PVSource_sp, QWidget* parent = 0);

	void add_view_display(QWidget* view_display, const QString& name);

public slots:
	void switch_with_central_widget(PVViewDisplay* display_dock = nullptr)
	{
		if (!display_dock) {
			display_dock = (PVViewDisplay*) sender()->parent();
		}
		QWidget* display = display_dock->widget();
		display->setParent(nullptr);

		QWidget* central_widget = centralWidget();
		central_widget->setParent(display_dock);

		setCentralWidget(display);
		display_dock->setWidget(central_widget);
	}

	void create_parallel_view()
	{
		QAction* action = (QAction*) sender();
		QVariant var = action->data();
		Picviz::PVView* view = var.value<Picviz::PVView*>();

		PVParallelView::PVLibView* parallel_lib_view;

		PVCore::PVProgressBox* pbox_lib = new PVCore::PVProgressBox("Creating new view...", (QWidget*) this);
		pbox_lib->set_enable_cancel(false);
		PVCore::PVProgressBox::progress<PVParallelView::PVLibView*>(boost::bind(&PVParallelView::common::get_lib_view, boost::ref(*view)), pbox_lib, parallel_lib_view);

		PVParallelView::PVFullParallelView* parallel_view = parallel_lib_view->create_view();
		connect(parallel_view, SIGNAL(new_zoomed_parallel_view(Picviz::PVView*, int)), this, SLOT(create_zoomed_parallel_view(Picviz::PVView*, int)));

		add_view_display(parallel_view, "Parallel view [" + view->get_name() + "]");
	}

	void create_listing_view()
	{
		QAction* action = (QAction*) sender();
		QVariant var = action->data();
		Picviz::PVView* view = var.value<Picviz::PVView*>();

		Picviz::PVView_p view_p = view->shared_from_this();
		PVListingModel* listing_model = new PVGuiQt::PVListingModel(view_p);
		PVListingSortFilterProxyModel* proxy_model = new PVGuiQt::PVListingSortFilterProxyModel(view_p);
		proxy_model->setSourceModel(listing_model);
		PVListingView* listing_view = new PVGuiQt::PVListingView(view_p);
		listing_view->setModel(proxy_model);

		add_view_display(listing_view, "Listing [" + view->get_name() + "]");
	}

public slots:
	void create_zoomed_parallel_view(Picviz::PVView* view, int axis_id)
	{
		QWidget* zoomed_parallel_view = PVParallelView::common::get_lib_view(*view)->create_zoomed_view(axis_id);

		add_view_display(zoomed_parallel_view, "Zoomed parallel view [" + view->get_name() + "]");
	}

	void create_datatree_view(bool create)
	{
		if (create) {
			PVRootTreeModel* datatree_model = new PVRootTreeModel(*_source);
			PVRootTreeView* data_tree_display = new PVRootTreeView(datatree_model);
			connect(data_tree_display, SIGNAL(destroyed(QObject*)), this, SLOT(uncheck_datatree_button()));
			add_view_display(data_tree_display, "Data tree");
		}
		else {
			for (auto display : _displays) {
				if (dynamic_cast<PVRootTreeView*>(display->widget())) {
					removeDockWidget(display);
				}
			}
		}
	}

	void uncheck_datatree_button()
	{
		_datatree_view_action->setChecked(false);
	}

private:
	QList<PVViewDisplay*> _displays;
	Picviz::PVSource_sp _source;
	QToolBar* _toolbar;
	QAction* _datatree_view_action;
};

class SlotHandler : public QObject
{
	Q_OBJECT;
public:
	SlotHandler(
		PVWorkspace* workspace,
		PVViewDisplay* view_display_to_switch
	) : _workspace(workspace), _view_display(view_display_to_switch) {}

public slots:
	void switch_displays()
	{
		_workspace->switch_with_central_widget(_view_display);
	}

private:
	PVWorkspace* _workspace;
	PVViewDisplay* _view_display;
};

}

#endif /* __PVGUIQT_PVWORKSPACE_H__ */
