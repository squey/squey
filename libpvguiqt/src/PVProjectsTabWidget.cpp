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
	setObjectName("PVProjectsTabWidget");
	setTabsClosable(true);
	connect(this, SIGNAL(currentChanged(int)), this, SLOT(current_tab_changed(int)));

	connect(tabBar(), SIGNAL(tabCloseRequested(int)), this, SLOT(tab_close_requested(int)));

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
	connect(workspace_tab_widget, SIGNAL(is_empty()), this, SLOT(close_project()));
	connect(workspace_tab_widget, SIGNAL(project_modified(bool)), this, SLOT(project_modified(bool)));

	insertTab(count(), workspace_tab_widget, scene_p->get_name());
	setCurrentIndex(count()-1);

	return workspace_tab_widget;
}

void PVGuiQt::PVProjectsTabWidget::project_modified(bool modified)
{
	PVWorkspacesTabWidget* workspace_tab_widget = (PVWorkspacesTabWidget*) sender();
	assert(workspace_tab_widget);
	int index = indexOf(workspace_tab_widget);
	QString text = tabText(index);
	const QString star = "*";
	if (modified && !text.endsWith(star)) {
		setTabText(index, text + "*");
	}
	else if (!modified && text.endsWith(star)) {
		text.truncate(text.size()-2);
		setTabText(index, text);
	}
}

bool PVGuiQt::PVProjectsTabWidget::save_modified_projects()
{
	for (int i = 2; i < count(); i++) {
		PVWorkspacesTabWidget* tab_widget = (PVWorkspacesTabWidget*) widget(i);
		if (tab_widget->is_project_modified()) {
			if (!tab_close_requested(i)) {
				return false;
			}
		}
	}

	return true;
}

void PVGuiQt::PVProjectsTabWidget::close_project()
{
	PVWorkspacesTabWidget* workspace_tab_widget = (PVWorkspacesTabWidget*) sender();
	assert(workspace_tab_widget);
	int index = indexOf(workspace_tab_widget);
	remove_project(index);
}

bool PVGuiQt::PVProjectsTabWidget::tab_close_requested(int index)
{
	if (maybe_save_project(index)) {
		remove_project(index);
		return true;
	}

	return false;
}

bool PVGuiQt::PVProjectsTabWidget::maybe_save_project(int index)
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	PVWorkspacesTabWidget* tab_widget = (PVWorkspacesTabWidget*) widget(index);
	if (tab_widget->is_project_modified()) {
		QMessageBox::StandardButton ret;
		QString project_name = tabText(index).left(tabText(index).size()-1);
		ret = QMessageBox::warning(this, tr("%1").arg(project_name),
				tr("The project \"%1\"has been modified.\n"
					"Do you want to save your changes?").arg(project_name),
				QMessageBox::Save | QMessageBox::Discard
				| QMessageBox::Cancel);
		if (ret == QMessageBox::Save) {
			return /*project_save_Slot()*/ true;
		}
		if (ret == QMessageBox::Discard) {
			return true;
		}
		else if (ret == QMessageBox::Cancel) {
			return false;
		}
	}
	return true;
#else
	return false;
#endif
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

void PVGuiQt::PVProjectsTabWidget::current_tab_changed(int index)
{
	if (index >= 2) {
		_current_project_index = index;
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
