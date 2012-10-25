/**
 * \file PVProjectsTabWidget.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <pvguiqt/PVProjectsTabWidget.h>
#include <pvguiqt/PVStartScreenWidget.h>

#include <QHBoxLayout>

PVGuiQt::PVProjectsTabWidget::PVProjectsTabWidget(QWidget* parent /*= 0*/) : QWidget(parent)
{
	_tab_bar = new QTabBar();
	_tab_bar->setShape(QTabBar::RoundedWest);
	_tab_bar->setTabsClosable(true);

	_stacked_widget = new QStackedWidget();

	QHBoxLayout* main_layout = new QHBoxLayout();

	_splitter = new __impl::PVSplitter(Qt::Horizontal);
	_splitter->setChildrenCollapsible(true);
	_splitter->addWidget(_tab_bar);
	_splitter->addWidget(_stacked_widget);
	_splitter->setStretchFactor(0, 0);
	_splitter->setStretchFactor(1, 1);
	QList<int> sizes;
	sizes << 1 << 2;
	_splitter->setSizes(sizes);

	main_layout->addWidget(_splitter);

	setLayout(main_layout);

	// Map QTabBar signal to QStackedWidget to keep the sync
	connect(_tab_bar, SIGNAL(currentChanged(int)), _stacked_widget, SLOT(setCurrentIndex(int)));

	create_unclosable_tabs();
}

void  PVGuiQt::PVProjectsTabWidget::create_unclosable_tabs()
{
	// Start screen widget
	_start_screen_widget = new PVGuiQt::PVStartScreenWidget();
	_tab_bar->addTab("Start screen");
	_tab_bar->tabButton(0, QTabBar::RightSide)->resize(0, 0);
	_tab_bar->setTabIcon(0, QIcon(":/picviz"));
	_stacked_widget->addWidget(_start_screen_widget);

	// Open workspaces
	_workspaces_tab_widget = new PVWorkspacesTabWidget();
	_tab_bar->addTab("Workspaces");
	_tab_bar->tabButton(1, QTabBar::RightSide)->resize(0, 0);
	_stacked_widget->addWidget(_workspaces_tab_widget);

	((__impl::PVSplitterHandle*) _splitter->handle(1))->set_max_size(_tab_bar->tabRect(0).width());
}

void PVGuiQt::PVProjectsTabWidget::collapse_tabs(bool collapse /* = true */)
{
	int max_size = ((__impl::PVSplitterHandle*) _splitter->handle(1))->get_max_size();
	QList<int> sizes;
	sizes << (collapse ? 0 : max_size) << 1;
	_splitter->setSizes(sizes);
}

PVGuiQt::PVWorkspacesTabWidget* PVGuiQt::PVProjectsTabWidget::add_project(Picviz::PVScene* scene, const QString& text)
{
	PVWorkspacesTabWidget* workspace_tab_widget = new PVWorkspacesTabWidget(scene);
	connect(workspace_tab_widget, SIGNAL(workspace_dragged_outside(QWidget*)), this, SLOT(emit_workspace_dragged_outside(QWidget*)));
	connect(workspace_tab_widget, SIGNAL(is_empty()), this, SLOT(close_project()));

	int index = _tab_bar->count() -1;
	_tab_bar->insertTab(index, text);
	_stacked_widget->insertWidget(index, workspace_tab_widget);

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
	_tab_bar->setCurrentIndex(_stacked_widget->indexOf(workspace_tab_widget));
}

void PVGuiQt::PVProjectsTabWidget::remove_workspace(PVWorkspace* workspace, bool animation /* = true */)
{
	Picviz::PVScene* scene = workspace->get_source()->get_parent<Picviz::PVScene>();
	PVWorkspacesTabWidget* workspace_tab_widget = get_workspace_tab_widget_from_scene(scene);
	workspace_tab_widget->remove_workspace(workspace_tab_widget->indexOf(workspace), animation);
}

void PVGuiQt::PVProjectsTabWidget::remove_project(PVWorkspacesTabWidget* workspace_tab_widget)
{
	int project_index = _stacked_widget->indexOf(workspace_tab_widget);
	if (project_index != -1) {
		_tab_bar->removeTab(project_index);
		_stacked_widget->removeWidget(workspace_tab_widget);

		if (_stacked_widget->count() == 2) {
			emit is_empty();
		}
	}
}

PVGuiQt::PVWorkspacesTabWidget* PVGuiQt::PVProjectsTabWidget::get_workspace_tab_widget_from_scene(const Picviz::PVScene* scene)
{
	for (int i = 0 ; i < _stacked_widget->count(); i++) {
		PVWorkspacesTabWidget* workspace_tab_widget = (PVWorkspacesTabWidget*) _stacked_widget->widget(i);
		if (workspace_tab_widget->get_scene() == scene) {
			return workspace_tab_widget;
		}
	}
	return nullptr;
}
