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

	void add_view_display(QWidget* view_display, const QString& name, bool can_be_central_display = true);
	void set_central_display(QWidget* view_widget, const QString& name);

public slots:
	void switch_with_central_widget(PVViewDisplay* display_dock = nullptr)
	{
		if (!display_dock) {
			display_dock = (PVViewDisplay*) sender()->parent();
		}
		QWidget* display_widget = display_dock->widget();

		PVViewDisplay* central_dock = (PVViewDisplay*) centralWidget();
		QWidget* central_widget = central_dock->widget();

		// Exchange widgets
		central_dock->setWidget(display_widget);
		display_dock->setWidget(central_widget);

		// Exchange titles
		QString central_title = central_dock->windowTitle();
		central_dock->setWindowTitle(display_dock->windowTitle());
		display_dock->setWindowTitle(central_title);
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

	void show_datatree_view(bool show)
	{
		for (auto display : _displays) {
			if (dynamic_cast<PVRootTreeView*>(display->widget())) {
				display->setVisible(show);
			}
		}
	}

	void check_datatree_button(bool check = false)
	{
		_datatree_view_action->setChecked(check);
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
