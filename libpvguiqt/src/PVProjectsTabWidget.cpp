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

	setTabsClosable(true);
	setObjectName("PVProjectsTabWidget");

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
	connect(_start_screen_widget, SIGNAL(new_format()), this, SIGNAL(new_format()));
	connect(_start_screen_widget, SIGNAL(load_format()), this, SIGNAL(load_format()));
	connect(_start_screen_widget, SIGNAL(load_project_from_path(const QString &)), this, SIGNAL(load_project_from_path(const QString &)));
	connect(_start_screen_widget, SIGNAL(edit_format(const QString &)), this, SIGNAL(edit_format(const QString &)));

	// Open workspaces
	_workspaces_tab_widget = new PVWorkspacesTabWidget();
	addTab(_workspaces_tab_widget, "");
	tabBar()->tabButton(1, QTabBar::RightSide)->resize(0, 0);
	setTabToolTip(1, "Workspaces");
	setTabIcon(1, QIcon(":/brush.png"));
}

PVGuiQt::PVWorkspacesTabWidget* PVGuiQt::PVProjectsTabWidget::add_project(Picviz::PVScene* scene, const QString& text)
{
	PVWorkspacesTabWidget* workspace_tab_widget = new PVWorkspacesTabWidget(scene);
	connect(workspace_tab_widget, SIGNAL(workspace_dragged_outside(QWidget*)), this, SLOT(emit_workspace_dragged_outside(QWidget*)));
	connect(workspace_tab_widget, SIGNAL(is_empty()), this, SLOT(close_project()));

	insertTab(count(), workspace_tab_widget, text);

	return workspace_tab_widget;
}

void PVGuiQt::PVProjectsTabWidget::close_project()
{
	PVWorkspacesTabWidget* workspace_tab_widget = (PVWorkspacesTabWidget*) sender();
	assert(workspace_tab_widget);
	remove_project(workspace_tab_widget);
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
		workspace_tab_widget = add_project(scene, scene->get_name());
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
	int project_index = indexOf(workspace_tab_widget);
	if (project_index != -1) {
		removeTab(project_index);

		if (count() == 2) {
			emit is_empty();
		}
	}
}

PVGuiQt::PVWorkspacesTabWidget* PVGuiQt::PVProjectsTabWidget::get_workspace_tab_widget_from_scene(const Picviz::PVScene* scene)
{
	for (int i = 0 ; i < count(); i++) {
		PVWorkspacesTabWidget* workspace_tab_widget = (PVWorkspacesTabWidget*) widget(i);
		if (workspace_tab_widget->get_scene() == scene) {
			return workspace_tab_widget;
		}
	}
	return nullptr;
}
