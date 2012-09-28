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

#include <pvguiqt/PVViewDisplay.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVLibView.h>

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

		QWidget* parallel_view = parallel_lib_view->create_view();

		add_view_display(parallel_view, "Test");
	}


private:
	QList<PVViewDisplay*> _displays;
	Picviz::PVSource_sp _source;
	QToolBar* _toolbar;
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
