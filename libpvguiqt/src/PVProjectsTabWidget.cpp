/**
 * \file PVProjectsTabWidget.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <pvguiqt/PVProjectsTabWidget.h>

#include <pvguiqt/PVWorkspace.h>

PVGuiQt::PVProjectsTabWidget::PVProjectsTabWidget(QWidget* parent) : QWidget(parent)
{
	_tab_bar = new QTabBar();
	_tab_bar->setShape(QTabBar::RoundedWest);
	_tab_bar->setTabsClosable(true);

	_stacked_widget = new QStackedWidget();

	_splitter = new __impl::PVSplitter(Qt::Horizontal, parent);
	_splitter->setChildrenCollapsible(true);
	_splitter->addWidget(_tab_bar);
	_splitter->addWidget(_stacked_widget);
	_splitter->resize(parent->size());
	_splitter->setStretchFactor(0, 0);
	_splitter->setStretchFactor(1, 1);
	QList<int> sizes;
	sizes << 1 << 2;
	_splitter->setSizes(sizes);

	// Map QTabBar signal to QStackedWidget to keep the sync
	connect(_tab_bar, SIGNAL(currentChanged(int)), _stacked_widget, SLOT(setCurrentIndex(int)));
}

void PVGuiQt::PVProjectsTabWidget::collapse_tabs(bool collapse /* true */)
{
	int max_size = ((__impl::PVSplitterHandle*) _splitter->handle(1))->get_max_size();
	QList<int> sizes;
	sizes << (collapse ? 0 : max_size) << 1;
	_splitter->setSizes(sizes);
}

PVGuiQt::PVWorkspacesTabWidget* PVGuiQt::PVProjectsTabWidget::add_project(Picviz::PVScene* scene, const QString& text)
{
	PVGuiQt::PVWorkspacesTabWidget* workspace_tab_widget = new PVGuiQt::PVWorkspacesTabWidget(scene);

	_tab_bar->addTab(text);
	_stacked_widget->addWidget(workspace_tab_widget);

	((__impl::PVSplitterHandle*) _splitter->handle(1))->set_max_size(_tab_bar->tabRect(0).width());

	return workspace_tab_widget;
}

void PVGuiQt::PVProjectsTabWidget::add_source(Picviz::PVSource* source)
{
	PVGuiQt::PVWorkspace* workspace = new PVGuiQt::PVWorkspace(source);

	Picviz::PVScene* scene = workspace->get_source()->get_parent<Picviz::PVScene>();
	PVWorkspacesTabWidget* workspace_tab_widget = get_workspace_tab_widget_from_scene(scene);

	if (workspace_tab_widget == nullptr) {
		workspace_tab_widget = add_project(scene, scene->get_name());
	}

	workspace_tab_widget->addTab(workspace, source->get_name());
}

PVGuiQt::PVWorkspacesTabWidget* PVGuiQt::PVProjectsTabWidget::get_workspace_tab_widget_from_scene(Picviz::PVScene* scene)
{
	for (int i = 0 ; i < _stacked_widget->count(); i++) {
		PVWorkspacesTabWidget* workspace_tab_widget = (PVWorkspacesTabWidget*) _stacked_widget->widget(i);
		if (workspace_tab_widget->get_scene() == scene) {
			return workspace_tab_widget;
		}
	}
	return nullptr;
}
