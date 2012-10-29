/**
 * \file PVProjectsTabWidget.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <pvguiqt/PVProjectsTabWidget.h>
#include <pvguiqt/PVStartScreenWidget.h>

#include <QHBoxLayout>

PVGuiQt::PVProjectsTabWidget::PVProjectsTabWidget(QWidget* parent /*= 0*/) : QTabWidget(parent)
{
	//_tab_bar->setShape(QTabBar::RoundedWest);
	//_tab_bar->setDocumentMode(true);
	//_tab_bar->setDrawBase(true);

	setObjectName("PVProjectsTabWidget");
	setTabsClosable(true);
	connect(this, SIGNAL(currentChanged(int)), this, SLOT(currentChanged_Slot(int)));

	connect(tabBar(), SIGNAL(tabCloseRequested(int)), this, SLOT(tabCloseRequested_Slot(int)));

	create_unclosable_tabs();
}

void  PVGuiQt::PVProjectsTabWidget::create_unclosable_tabs()
{
	// Start screen widget
	_start_screen_widget = new PVGuiQt::PVStartScreenWidget();
	addTab(_start_screen_widget, "");
	setTabPosition(QTabWidget::West);
	tabBar()->tabButton(0, QTabBar::RightSide)->resize(0, 0);
	setTabIcon(0, QIcon(":/picviz"));
	setTabToolTip(0, "Start screen");
	connect(_start_screen_widget, SIGNAL(load_source_from_description(PVRush::PVSourceDescription)), this, SIGNAL(load_source_from_description(PVRush::PVSourceDescription)));
	connect(_start_screen_widget, SIGNAL(new_project()), this, SIGNAL(new_project()));
	connect(_start_screen_widget, SIGNAL(load_project()), this, SIGNAL(load_project()));
	connect(_start_screen_widget, SIGNAL(load_project_from_path(const QString &)), this, SIGNAL(load_project_from_path(const QString &)));
	connect(_start_screen_widget, SIGNAL(new_format()), this, SIGNAL(new_format()));
	connect(_start_screen_widget, SIGNAL(load_format()), this, SIGNAL(load_format()));
	connect(_start_screen_widget, SIGNAL(edit_format(const QString &)), this, SIGNAL(edit_format(const QString &)));

	// Open workspaces
	_workspaces_tab_widget = new PVWorkspacesTabWidget();
	addTab(_workspaces_tab_widget, "");
	tabBar()->tabButton(1, QTabBar::RightSide)->resize(0, 0);
	setTabToolTip(1, "Workspaces");
	setTabIcon(1, QIcon(":/brush.png"));
}

PVGuiQt::PVWorkspacesTabWidget* PVGuiQt::PVProjectsTabWidget::add_project(Picviz::PVScene_p scene_p)
{
	PVWorkspacesTabWidget* workspace_tab_widget = new PVWorkspacesTabWidget(scene_p);
	connect(workspace_tab_widget, SIGNAL(workspace_dragged_outside(QWidget*)), this, SLOT(emit_workspace_dragged_outside(QWidget*)));
	connect(workspace_tab_widget, SIGNAL(is_empty()), this, SLOT(close_project_Slot()));

	insertTab(count(), workspace_tab_widget, scene_p->get_name());
	setCurrentIndex(count()-1);

	return workspace_tab_widget;
}

void PVGuiQt::PVProjectsTabWidget::close_project_Slot()
{
	PVWorkspacesTabWidget* workspace_tab_widget = (PVWorkspacesTabWidget*) sender();
	assert(workspace_tab_widget);
	tabCloseRequested_Slot(indexOf(workspace_tab_widget));
}

void PVGuiQt::PVProjectsTabWidget::tabCloseRequested_Slot(int index)
{
	remove_project(index);
}

PVGuiQt::PVWorkspace* PVGuiQt::PVProjectsTabWidget::add_source(Picviz::PVSource* source)
{
	PVGuiQt::PVWorkspace* workspace = new PVGuiQt::PVWorkspace(source);

	add_workspace(workspace);

	return workspace;
}

void PVGuiQt::PVProjectsTabWidget::add_workspace(PVWorkspace* workspace)
{
	Picviz::PVScene* scene = workspace->get_source()->get_parent<Picviz::PVScene>();
	PVWorkspacesTabWidget* workspace_tab_widget = get_workspace_tab_widget_from_scene(scene);

	if (!workspace_tab_widget) {
		workspace_tab_widget = add_project(scene->shared_from_this());
	}

	workspace_tab_widget->addTab(workspace, workspace->get_source()->get_name());
	setCurrentIndex(indexOf(workspace_tab_widget));
}

void PVGuiQt::PVProjectsTabWidget::remove_workspace(PVWorkspace* workspace, bool animation /* = true */)
{
	Picviz::PVScene* scene = workspace->get_source()->get_parent<Picviz::PVScene>();
	PVWorkspacesTabWidget* workspace_tab_widget = get_workspace_tab_widget_from_scene(scene);
	workspace_tab_widget->remove_workspace(workspace_tab_widget->indexOf(workspace), animation);
}

void PVGuiQt::PVProjectsTabWidget::remove_project(PVWorkspacesTabWidget* workspace_tab_widget)
{
	remove_project(indexOf(workspace_tab_widget));
}

void PVGuiQt::PVProjectsTabWidget::currentChanged_Slot(int index)
{
	if (index >= 2) {
		_current_project_index = index;
	}
}

void PVGuiQt::PVProjectsTabWidget::remove_project(int index)
{
	if (index != -1) {
		PVWorkspacesTabWidget* tab_widget = (PVWorkspacesTabWidget*) widget(index);
		removeTab(index);
		tab_widget->deleteLater();

		if (count() == 2) {
			setCurrentIndex(0);
			emit is_empty();
		}
	}
}

PVGuiQt::PVWorkspacesTabWidget* PVGuiQt::PVProjectsTabWidget::get_workspace_tab_widget_from_scene(const Picviz::PVScene* scene)
{
	for (int i = 2 ; i < count(); i++) {
		PVWorkspacesTabWidget* workspace_tab_widget = (PVWorkspacesTabWidget*) widget(i);
		if (workspace_tab_widget->get_scene() == scene) {
			return workspace_tab_widget;
		}
	}
	return nullptr;
}

Picviz::PVScene* PVGuiQt::PVProjectsTabWidget::get_scene_from_path(const QString & path)
{
	for (int i = 2 ; i < count(); i++) {
		Picviz::PVScene* scene = ((PVWorkspacesTabWidget* ) widget(i))->get_scene();
		if (scene->get_path() == path) {
			return scene;
		}
	}

	return nullptr;
}
